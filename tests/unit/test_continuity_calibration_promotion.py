"""Evidence gates for continuity coefficient calibration and Batch-A promotion."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.physiology_synergy.calibrate_continuity_reward import (
    BASELINE_ROLLOUT_SCHEMA_VERSION,
    PENALTY_BUDGET_FRACTIONS,
    baseline_rollout_fingerprint,
    build_baseline_rollout_evidence,
    build_continuity_reward_calibration,
    calibration_fingerprint,
    validate_continuity_reward_calibration,
)
from analysis.physiology_synergy.collect_continuity_baseline import (
    _TRAJECTORY_ARRAY_FIELDS,
    CONTINUITY_BASELINE_ENVIRONMENT_SCHEMA_VERSION,
    CONTINUITY_BASELINE_HELDOUT_SCHEMA_VERSION,
    CONTINUITY_BASELINE_ROLLOUT_MANIFEST_SCHEMA_VERSION,
    _trajectory_manifest_digest,
    rollout_manifest_fingerprint,
)
from analysis.physiology_synergy.promote_continuity_chains import (
    BATCH_A_CHAIN_IDS,
    CHAIN_REVIEW_SCHEMA_VERSION,
    chain_review_fingerprint,
    promote_batch_a_continuity_graph,
    validate_chain_review,
)
from musclemimic.physiology.anatomical_groups import load_anatomical_taxonomy
from musclemimic.physiology.continuity_groups import (
    load_fascicle_continuity_graph,
    resolve_fascicle_continuity_reward_gate,
    validate_fascicle_continuity_graph,
)

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_curated_v1.json"
GRAPH_PATH = ROOT / "configs/physiology/myofullbody_354_fascicle_continuity_v1.json"


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _rollout_manifest() -> dict:
    taxonomy = load_anatomical_taxonomy(TAXONOMY_PATH)
    graph = load_fascicle_continuity_graph(GRAPH_PATH, taxonomy=taxonomy)
    environment = {
        "schema_version": CONTINUITY_BASELINE_ENVIRONMENT_SCHEMA_VERSION,
        "resolved_env_params": {"disable_fingers": True},
        "resolved_task_factory": {"name": "fixture"},
        "resolved_action_representation": {
            "enabled": True,
            "mode": "fixed_synergy",
            "residual": {"enabled": False},
        },
        "runtime_model_hash": "1" * 64,
        "muscle_channel_core_fingerprint": "2" * 64,
        "ordered_actuator_schema_hash": "3" * 64,
        "ordered_action_dim": 354,
        "control_dt": 0.01,
        "horizon": 32,
    }
    environment["environment_fingerprint"] = _json_sha(environment)
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
        "ordered_motion_paths": ["heldout/motion"],
        "trajectory_count": 2,
        "trajectory_lengths": [32, 32],
        "frequency_hz": 100.0,
        "array_manifest": arrays,
        "trajectory_content_sha256": _trajectory_manifest_digest(arrays),
    }
    heldout["heldout_split_fingerprint"] = _json_sha(heldout)
    payload = {
        "schema_version": CONTINUITY_BASELINE_ROLLOUT_MANIFEST_SCHEMA_VERSION,
        "semantics": "evaluate_all_once_per_heldout_from_frame_zero_v1",
        "policy": {
            "run_id": "forehand_contdiag_baseline_s0",
            "action_mode": "fixed_synergy",
            "config_hash": "fixture-config-hash",
            "checkpoint_fingerprint": "a" * 64,
            "promotion_fingerprint": "b" * 64,
        },
        "physiology": {
            "taxonomy_fingerprint": taxonomy.fingerprint,
            "continuity_graph_fingerprint": graph.graph_fingerprint,
            "measured_chain_count": len(graph.chains),
            "measured_edge_count": graph.edge_count,
        },
        "environment_manifest": environment,
        "heldout_split_manifest": heldout,
        "protocol": {
            "backend": "mjx",
            "deterministic": True,
            "eval_seed": 0,
            "num_envs": 2,
            "evaluate_all": True,
            "start_from_beginning": True,
            "padding_steps_included": False,
            "post_completion_steps_included": False,
            "primary_reward_sample": "reward_imitation_total_before_penalties",
            "primary_continuity_sample": "post_transition_data.act_fascicle_continuity_loss",
        },
    }
    payload["rollout_manifest_fingerprint"] = rollout_manifest_fingerprint(payload)
    return payload


def _baseline() -> dict:
    taxonomy = load_anatomical_taxonomy(TAXONOMY_PATH)
    graph = load_fascicle_continuity_graph(GRAPH_PATH, taxonomy=taxonomy)
    manifest = _rollout_manifest()
    reward = np.linspace(0.5, 1.5, 64)
    loss = np.linspace(0.01, 0.20, 64)
    trajectory_index = np.repeat(np.arange(2), 32)
    trajectory_step = np.tile(np.arange(32), 2)
    payload = {
        "schema_version": BASELINE_ROLLOUT_SCHEMA_VERSION,
        "identity": {
            "run_id": "forehand_contdiag_baseline_s0",
            "action_mode": "fixed_synergy",
            "config_hash": "fixture-config-hash",
            "checkpoint_fingerprint": "a" * 64,
            "promotion_fingerprint": "b" * 64,
            "rollout_manifest_fingerprint": manifest["rollout_manifest_fingerprint"],
            "environment_fingerprint": manifest["environment_manifest"]["environment_fingerprint"],
            "heldout_split_fingerprint": manifest["heldout_split_manifest"]["heldout_split_fingerprint"],
            "taxonomy_fingerprint": taxonomy.fingerprint,
            "continuity_graph_fingerprint": graph.graph_fingerprint,
        },
        "coverage": {
            "trajectory_count": 2,
            "step_count": 64,
            "measured_chain_count": len(graph.chains),
            "measured_edge_count": graph.edge_count,
        },
        "samples": {
            "trajectory_index": trajectory_index.tolist(),
            "trajectory_step": trajectory_step.tolist(),
            "imitation_reward": reward.tolist(),
            "activation_continuity_loss": loss.tolist(),
            "measured_chain_count": [len(graph.chains)] * 64,
            "measured_edge_count": [graph.edge_count] * 64,
        },
    }
    payload["artifact_fingerprint"] = baseline_rollout_fingerprint(payload)
    return payload


def _calibrate(baseline: dict, *, selected_budget_fraction: float) -> dict:
    return build_continuity_reward_calibration(
        baseline,
        rollout_manifest=_rollout_manifest(),
        selected_budget_fraction=selected_budget_fraction,
    )


def _review(calibration: dict) -> dict:
    taxonomy = load_anatomical_taxonomy(TAXONOMY_PATH)
    graph = load_fascicle_continuity_graph(GRAPH_PATH, taxonomy=taxonomy)
    by_id = {chain["chain_id"]: chain for chain in graph.chains}
    payload = {
        "schema_version": CHAIN_REVIEW_SCHEMA_VERSION,
        "batch": "A",
        "source_graph_fingerprint": graph.graph_fingerprint,
        "taxonomy_fingerprint": taxonomy.fingerprint,
        "calibration_fingerprint": calibration["calibration_fingerprint"],
        "reviewer": {
            "name": "Independent anatomy reviewer",
            "affiliation_or_role": "biomechanics review fixture",
            "reviewed_at_utc": "2026-07-30T12:00:00Z",
            "independent_of_code_author": True,
        },
        "chains": [
            {
                "chain_id": chain_id,
                "checks": {
                    "exact_asset_topology_reviewed": True,
                    "same_side_verified": True,
                    "adjacent_level_definition_reviewed": True,
                    "baseline_activation_distribution_reviewed": True,
                    "deadband_data_supported": True,
                    "approve_training": True,
                },
                "approved_deadband": by_id[chain_id]["deadband"],
                "provenance": [
                    {
                        "kind": "independent_anatomical_review",
                        "reference": f"fixture-review:{chain_id}",
                    }
                ],
            }
            for chain_id in BATCH_A_CHAIN_IDS
        ],
    }
    payload["review_fingerprint"] = chain_review_fingerprint(payload)
    return payload


def _raw_step_samples() -> dict:
    trajectory_index = np.repeat(np.arange(2), 32)
    trajectory_step = np.tile(np.arange(32), 2)
    return {
        "trajectory_index": trajectory_index.tolist(),
        "trajectory_step": trajectory_step.tolist(),
        "reward_imitation_total": np.linspace(0.5, 1.5, 64).tolist(),
        "fascicle_continuity_loss": np.linspace(0.01, 0.20, 64).tolist(),
        "fascicle_continuity_measured_chain_count": [28.0] * 64,
        "fascicle_continuity_measured_edge_count": [140.0] * 64,
    }


def test_baseline_builder_derives_coverage_from_unreduced_evaluate_all_steps():
    identity = _baseline()["identity"]
    evidence = build_baseline_rollout_evidence(
        identity,
        _raw_step_samples(),
        expected_trajectory_count=2,
        expected_measured_chain_count=28,
        expected_measured_edge_count=140,
    )

    assert evidence["coverage"] == {
        "trajectory_count": 2,
        "step_count": 64,
        "measured_chain_count": 28,
        "measured_edge_count": 140,
    }
    assert evidence["samples"]["imitation_reward"][0] == pytest.approx(0.5)
    assert evidence["artifact_fingerprint"] == baseline_rollout_fingerprint(evidence)


def test_baseline_builder_rejects_padding_duplicates_and_runtime_coverage_drift():
    identity = _baseline()["identity"]

    duplicate = _raw_step_samples()
    duplicate["trajectory_step"][-1] = 30
    with pytest.raises(ValueError, match="duplicate trajectory steps"):
        build_baseline_rollout_evidence(
            identity,
            duplicate,
            expected_trajectory_count=2,
            expected_measured_chain_count=28,
            expected_measured_edge_count=140,
        )

    skipped = _raw_step_samples()
    skipped["trajectory_step"][33] = 99
    with pytest.raises(ValueError, match="not one ordered prefix"):
        build_baseline_rollout_evidence(
            identity,
            skipped,
            expected_trajectory_count=2,
            expected_measured_chain_count=28,
            expected_measured_edge_count=140,
        )

    drift = _raw_step_samples()
    drift["fascicle_continuity_measured_edge_count"][10] = 139.0
    with pytest.raises(ValueError, match="measured-edge coverage differs"):
        build_baseline_rollout_evidence(
            identity,
            drift,
            expected_trajectory_count=2,
            expected_measured_chain_count=28,
            expected_measured_edge_count=140,
        )

    missing_trajectory = _raw_step_samples()
    keep = np.asarray(missing_trajectory["trajectory_index"]) == 0
    missing_trajectory = {key: np.asarray(value)[keep].tolist() for key, value in missing_trajectory.items()}
    with pytest.raises(ValueError, match="trajectory coverage differs"):
        build_baseline_rollout_evidence(
            identity,
            missing_trajectory,
            expected_trajectory_count=2,
            expected_measured_chain_count=28,
            expected_measured_edge_count=140,
        )


def test_calibration_uses_raw_heldout_quantiles_and_preregistered_budgets():
    baseline = _baseline()
    calibration = _calibrate(
        baseline,
        selected_budget_fraction=0.01,
    )

    median_reward = np.quantile(baseline["samples"]["imitation_reward"], 0.5)
    q95_loss = np.quantile(baseline["samples"]["activation_continuity_loss"], 0.95)
    assert calibration["statistics"]["median_imitation_reward"] == pytest.approx(median_reward)
    assert calibration["statistics"]["q95_activation_continuity_loss"] == pytest.approx(q95_loss)
    assert [item["fraction_of_median_imitation_reward"] for item in calibration["candidate_budgets"]] == list(
        PENALTY_BUDGET_FRACTIONS
    )
    assert calibration["selection"]["coefficient"] == pytest.approx(0.01 * median_reward / q95_loss)
    assert calibration["selection"]["dynamic_validation_reweighting_allowed"] is False
    assert validate_continuity_reward_calibration(calibration) == calibration


def test_calibration_rejects_stale_or_insufficient_baseline_evidence():
    stale = _baseline()
    stale["samples"]["activation_continuity_loss"][0] += 0.01
    with pytest.raises(ValueError, match="fingerprint is stale"):
        _calibrate(stale, selected_budget_fraction=0.01)

    short = _baseline()
    short["coverage"]["trajectory_count"] = 1
    short["coverage"]["step_count"] = 8
    short["samples"] = {key: values[:8] for key, values in short["samples"].items()}
    short["artifact_fingerprint"] = baseline_rollout_fingerprint(short)
    with pytest.raises(ValueError, match="at least 32 held-out steps"):
        _calibrate(short, selected_budget_fraction=0.01)


def test_calibration_revalidates_persisted_coordinates_and_rollout_binding():
    reordered = _baseline()
    reordered["samples"]["trajectory_step"][1] = 9
    reordered["artifact_fingerprint"] = baseline_rollout_fingerprint(reordered)
    with pytest.raises(ValueError, match=r"duplicate trajectory steps|not one ordered prefix"):
        _calibrate(reordered, selected_budget_fraction=0.01)

    manifest = _rollout_manifest()
    manifest["policy"]["run_id"] = "another-policy"
    manifest["rollout_manifest_fingerprint"] = rollout_manifest_fingerprint(manifest)
    with pytest.raises(ValueError, match="identity differs"):
        build_continuity_reward_calibration(
            _baseline(),
            rollout_manifest=manifest,
            selected_budget_fraction=0.01,
        )


def test_batch_a_promotion_creates_new_graph_and_enables_only_reviewed_chains():
    calibration = _calibrate(
        _baseline(),
        selected_budget_fraction=0.01,
    )
    review = _review(calibration)
    promoted_payload = promote_batch_a_continuity_graph(
        taxonomy_path=TAXONOMY_PATH,
        source_graph_path=GRAPH_PATH,
        calibration=calibration,
        review=review,
    )
    taxonomy = load_anatomical_taxonomy(TAXONOMY_PATH)
    source = load_fascicle_continuity_graph(GRAPH_PATH, taxonomy=taxonomy)
    promoted = validate_fascicle_continuity_graph(promoted_payload, taxonomy=taxonomy)
    enabled = tuple(chain["chain_id"] for chain in promoted.chains if chain["training_enabled"])

    assert promoted.graph_fingerprint != source.graph_fingerprint
    assert enabled == BATCH_A_CHAIN_IDS
    assert all(
        chain["review_status"] == "provisional" and not chain["training_enabled"]
        for chain in promoted.chains
        if chain["chain_id"] not in BATCH_A_CHAIN_IDS
    )
    assert resolve_fascicle_continuity_reward_gate(promoted, enabled=True)[0] is True
    promotion = promoted.generation["training_promotion"]
    assert promotion["calibration_fingerprint"] == calibration["calibration_fingerprint"]
    assert promotion["review_fingerprint"] == review["review_fingerprint"]


def test_batch_a_promotion_fails_closed_on_incomplete_or_cross_bound_review():
    calibration = _calibrate(
        _baseline(),
        selected_budget_fraction=0.005,
    )
    incomplete = _review(calibration)
    incomplete["chains"][0]["checks"]["deadband_data_supported"] = False
    incomplete["review_fingerprint"] = chain_review_fingerprint(incomplete)
    with pytest.raises(ValueError, match="incomplete promotion review"):
        validate_chain_review(incomplete)

    drifted = _review(calibration)
    drifted["source_graph_fingerprint"] = "9" * 64
    drifted["review_fingerprint"] = chain_review_fingerprint(drifted)
    with pytest.raises(ValueError, match="source graph differs"):
        promote_batch_a_continuity_graph(
            taxonomy_path=TAXONOMY_PATH,
            source_graph_path=GRAPH_PATH,
            calibration=calibration,
            review=drifted,
        )

    wrong_coverage = copy.deepcopy(calibration)
    wrong_coverage["coverage"]["measured_edge_count"] -= 1
    wrong_coverage["calibration_fingerprint"] = calibration_fingerprint(wrong_coverage)
    coverage_review = _review(wrong_coverage)
    with pytest.raises(ValueError, match="measured-edge coverage differs"):
        promote_batch_a_continuity_graph(
            taxonomy_path=TAXONOMY_PATH,
            source_graph_path=GRAPH_PATH,
            calibration=wrong_coverage,
            review=coverage_review,
        )


def test_chain_review_requires_an_explicit_utc_timestamp():
    calibration = _calibrate(
        _baseline(),
        selected_budget_fraction=0.01,
    )
    review = _review(calibration)
    review["reviewer"]["reviewed_at_utc"] = "2026-07-30T12:00:00"
    review["review_fingerprint"] = chain_review_fingerprint(review)
    with pytest.raises(ValueError, match="explicit UTC offset"):
        validate_chain_review(review)


def test_calibration_and_review_fingerprints_detect_output_tampering():
    calibration = _calibrate(
        _baseline(),
        selected_budget_fraction=0.02,
    )
    tampered_calibration = copy.deepcopy(calibration)
    tampered_calibration["selection"]["coefficient"] *= 2.0
    with pytest.raises(ValueError, match="selected coefficient is stale"):
        validate_continuity_reward_calibration(tampered_calibration)

    review = _review(calibration)
    review["reviewer"]["name"] = "Changed reviewer"
    with pytest.raises(ValueError, match="review fingerprint is stale"):
        validate_chain_review(review)
