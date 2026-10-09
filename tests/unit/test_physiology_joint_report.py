"""Fail-closed bindings for the physiology/synergy/EMG joint report."""

from __future__ import annotations

import copy

import pytest

from analysis.physiology_synergy.build_joint_report import (
    JOINT_REPORT_SCHEMA_VERSION,
    build_joint_report,
    build_rollout_metrics_evidence,
    joint_report_fingerprint,
    rollout_metrics_evidence_fingerprint,
)
from musclemimic.runner.checkpointing import config_hash

CHECKPOINT = "a" * 64
PROMOTION = "b" * 64
BASIS = "c" * 64
TAXONOMY = "d" * 64
GRAPH = "e" * 64
DATASET = "f" * 64
EVENT = "1" * 64
MAPPING = "2" * 64
TRIAL = "3" * 64
EVALUATION_SPLIT = "6" * 64
ENVIRONMENT = "7" * 64


def _inputs(*, with_emg: bool = True) -> dict:
    physiology = {
        "schema_version": "simulation_physiology_report_v2",
        "lineage": {
            "policy_evidence": {
                "policy_checkpoint_fingerprint": CHECKPOINT,
                "policy_promotion_fingerprint": PROMOTION,
                "formal_synergy_basis_fingerprint": BASIS,
            },
            "event_reference_fingerprint": EVENT,
            "session_uid": "heldout-session",
        },
        "fascicle_continuity": {
            "graph_fingerprint": GRAPH,
            "taxonomy_binding": {"taxonomy_fingerprint": TAXONOMY},
            "signal_priority": {
                "primary": "muscle_activation",
                "secondary": "effective_muscle_excitation",
            },
            "coverage": {
                "declared_chain_count": 28,
                "measured_chain_count": 28,
                "training_enabled_chain_count": 0,
                "measured_edge_count": 140,
                "continuity_measured": True,
            },
            "activation": {"aggregate": {"loss": {"mean": 0.2}}},
            "excitation": {"aggregate": {"loss": {"mean": 0.3}}},
        },
    }
    emg = None
    if with_emg:
        emg = {
            "schema_version": "emg_synergy_report_v2",
            "claim_scope": "whole_body_15_of_16_surface_channels",
            "claim_limitations": ["descriptive held-out comparison"],
            "policy_evidence": {
                "policy_checkpoint_fingerprint": CHECKPOINT,
                "policy_promotion_fingerprint": PROMOTION,
                "formal_synergy_basis_fingerprint": BASIS,
            },
            "mapping": {"model_binding": {"taxonomy_fingerprint": TAXONOMY}},
            "input_fingerprints": {"mapping_json_sha256": MAPPING},
            "trial_binding": {
                "binding_fingerprint": TRIAL,
                "session_uids": ["heldout-session"],
                "reference_trial_fingerprints": [EVENT],
            },
            "synergy": {"rank": 4, "similarity": {"mean": 0.75}},
        }
    inputs = {
        "rollout_metrics": {},
        "physiology_report": physiology,
        "synergy_basis_manifest": {
            "artifact_fingerprint": BASIS,
            "source_dataset_fingerprint": DATASET,
            "split_provenance": {"train": ["trial-a"], "validation": ["trial-b"]},
        },
        "frozen_decoder_manifest": {
            "schema_version": "early_synergy_action_v2",
            "basis_fingerprint": BASIS,
            "mode": "fixed_synergy",
        },
        "resolved_config": {
            "experiment": {
                "env_params": {
                    "reward_params": {
                        "intra_muscle_consistency": {
                            "mode": "diagnostics",
                            "coefficient": 0.0,
                            "expected_taxonomy_fingerprint": TAXONOMY,
                            "expected_continuity_fingerprint": GRAPH,
                        }
                    }
                }
            }
        },
        "checkpoint_evidence": {
            "run_id": "fixture-run",
            "config_hash": "pending",
            "checkpoint_fingerprint": CHECKPOINT,
            "promotion_fingerprint": PROMOTION,
            "branch_commit_sha": "4" * 40,
        },
        "branch_commit_sha": "4" * 40,
        "emg_report": emg,
    }
    inputs["checkpoint_evidence"]["config_hash"] = config_hash(inputs["resolved_config"]["experiment"])
    inputs["rollout_metrics"] = build_rollout_metrics_evidence(
        identity={
            "run_id": inputs["checkpoint_evidence"]["run_id"],
            "config_hash": inputs["checkpoint_evidence"]["config_hash"],
            "checkpoint_fingerprint": CHECKPOINT,
            "promotion_fingerprint": PROMOTION,
            "formal_synergy_basis_fingerprint": BASIS,
            "taxonomy_fingerprint": TAXONOMY,
            "continuity_graph_fingerprint": GRAPH,
            "event_reference_fingerprint": EVENT,
            "session_uid": "heldout-session",
            "evaluation_split_fingerprint": EVALUATION_SPLIT,
            "environment_fingerprint": ENVIRONMENT,
        },
        metrics={
            "mean_episode_return": 0.9,
            "early_termination_rate": 0.01,
            "err_rpos": 0.03,
            "synergy_coefficient_effective_dimension": 7.0,
            "fascicle_continuity_loss": 0.2,
        },
    )
    return inputs


def _rebind_rollout_to_config(inputs: dict) -> None:
    evidence = inputs["rollout_metrics"]
    identity = copy.deepcopy(evidence["identity"])
    identity["config_hash"] = inputs["checkpoint_evidence"]["config_hash"]
    inputs["rollout_metrics"] = build_rollout_metrics_evidence(
        identity=identity,
        metrics=evidence["metrics"],
    )


def test_joint_report_keeps_activation_excitation_and_all_required_bindings_separate():
    inputs = _inputs()
    report = build_joint_report(**inputs)

    assert report["schema_version"] == JOINT_REPORT_SCHEMA_VERSION
    assert report["report_fingerprint"] == joint_report_fingerprint(report)
    identity = report["identity"]
    assert identity["checkpoint_fingerprint"] == CHECKPOINT
    assert identity["promotion_fingerprint"] == PROMOTION
    assert identity["taxonomy_fingerprint"] == TAXONOMY
    assert identity["continuity_graph_fingerprint"] == GRAPH
    assert identity["formal_synergy_basis_fingerprint"] == BASIS
    assert identity["rollout_metrics_fingerprint"] == inputs["rollout_metrics"]["artifact_fingerprint"]
    assert identity["emg_mapping_fingerprint"] == MAPPING
    assert len(identity["frozen_decoder_fingerprint"]) == 64
    assert identity["data_evidence"]["event_reference_fingerprint"] == EVENT
    assert identity["data_evidence"]["emg_binding_fingerprint"] == TRIAL
    consistency = report["activation_consistency"]
    assert consistency["activation"] != consistency["excitation"]
    assert consistency["coverage"]["measured_edge_count"] == 140
    assert report["claim_scope"]["continuity"] == "simulation_activation_diagnostic_only"
    assert report["claim_scope"]["cross_space_causal_claim"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda inputs: inputs["physiology_report"]["lineage"]["policy_evidence"].update(
                policy_checkpoint_fingerprint="9" * 64
            ),
            "physiology checkpoint differs",
        ),
        (
            lambda inputs: inputs["frozen_decoder_manifest"].update(basis_fingerprint="9" * 64),
            "decoder formal synergy basis differs",
        ),
        (
            lambda inputs: inputs["emg_report"]["mapping"]["model_binding"].update(taxonomy_fingerprint="9" * 64),
            "EMG mapping taxonomy differs",
        ),
        (
            lambda inputs: inputs["physiology_report"]["fascicle_continuity"]["coverage"].update(measured_edge_count=0),
            "non-empty measured continuity coverage",
        ),
    ],
)
def test_joint_report_rejects_cross_artifact_identity_drift(mutator, message):
    inputs = _inputs()
    mutator(inputs)
    with pytest.raises(ValueError, match=message):
        build_joint_report(**inputs)


def test_joint_report_without_emg_explicitly_limits_claim_scope():
    inputs = _inputs(with_emg=False)
    report = build_joint_report(**inputs)

    assert report["identity"]["emg_mapping_fingerprint"] is None
    assert report["mapped_15ch_synergy"] == {"status": "not_provided"}
    assert report["emg_comparison"] == {"status": "not_provided"}
    assert report["claim_scope"]["emg"] == "not_measured"
    assert any("No EMG report" in value for value in report["limitations"])


def test_joint_report_rejects_nonzero_diagnostics_coefficient():
    inputs = copy.deepcopy(_inputs())
    continuity = inputs["resolved_config"]["experiment"]["env_params"]["reward_params"]["intra_muscle_consistency"]
    continuity["coefficient"] = 0.1
    inputs["checkpoint_evidence"]["config_hash"] = config_hash(inputs["resolved_config"]["experiment"])
    _rebind_rollout_to_config(inputs)
    with pytest.raises(ValueError, match="non-zero coefficient"):
        build_joint_report(**inputs)


def test_joint_report_reward_mode_binds_calibration_review_and_fixed_coefficient():
    inputs = _inputs(with_emg=False)
    promotion = {
        "batch": "A",
        "source_graph_fingerprint": "5" * 64,
        "calibration_fingerprint": "6" * 64,
        "review_fingerprint": "7" * 64,
        "selected_reward_coefficient": 0.01,
        "promoted_chain_ids": ["eo_r", "io_r", "eo_l", "io_l"],
    }
    continuity = inputs["physiology_report"]["fascicle_continuity"]
    continuity["coverage"]["training_enabled_chain_count"] = 4
    continuity["training_promotion"] = promotion
    config = inputs["resolved_config"]["experiment"]["env_params"]["reward_params"]["intra_muscle_consistency"]
    config.update(
        mode="reward",
        coefficient=0.01,
        expected_calibration_fingerprint="6" * 64,
    )
    inputs["checkpoint_evidence"]["config_hash"] = config_hash(inputs["resolved_config"]["experiment"])
    _rebind_rollout_to_config(inputs)

    report = build_joint_report(**inputs)

    assert report["identity"]["continuity_calibration_fingerprint"] == "6" * 64
    assert report["identity"]["continuity_review_fingerprint"] == "7" * 64
    assert report["activation_consistency"]["training_promotion"] == promotion
    assert report["claim_scope"]["continuity"] == "verified_training_prior_and_diagnostic"


def test_joint_report_rejects_resolved_config_and_reward_promotion_drift():
    inputs = _inputs(with_emg=False)
    inputs["resolved_config"]["experiment"]["total_timesteps"] = 123
    with pytest.raises(ValueError, match="config hash differs"):
        build_joint_report(**inputs)

    inputs = _inputs(with_emg=False)
    continuity = inputs["physiology_report"]["fascicle_continuity"]
    continuity["coverage"]["training_enabled_chain_count"] = 1
    continuity["training_promotion"] = {
        "batch": "A",
        "source_graph_fingerprint": "5" * 64,
        "calibration_fingerprint": "6" * 64,
        "review_fingerprint": "7" * 64,
        "selected_reward_coefficient": 0.02,
        "promoted_chain_ids": ["eo_r"],
    }
    config = inputs["resolved_config"]["experiment"]["env_params"]["reward_params"]["intra_muscle_consistency"]
    config.update(
        mode="reward",
        coefficient=0.01,
        expected_calibration_fingerprint="6" * 64,
    )
    inputs["checkpoint_evidence"]["config_hash"] = config_hash(inputs["resolved_config"]["experiment"])
    _rebind_rollout_to_config(inputs)
    with pytest.raises(ValueError, match="coefficient differs"):
        build_joint_report(**inputs)


def test_joint_report_rejects_stale_or_cross_policy_rollout_metrics():
    inputs = _inputs(with_emg=False)
    inputs["rollout_metrics"]["metrics"]["err_rpos"] = 9.0
    with pytest.raises(ValueError, match="rollout metrics evidence fingerprint is stale"):
        build_joint_report(**inputs)

    inputs = _inputs(with_emg=False)
    inputs["rollout_metrics"]["identity"]["checkpoint_fingerprint"] = "9" * 64
    inputs["rollout_metrics"]["artifact_fingerprint"] = rollout_metrics_evidence_fingerprint(inputs["rollout_metrics"])
    with pytest.raises(ValueError, match="rollout metrics checkpoint_fingerprint differs"):
        build_joint_report(**inputs)
