"""Regression gates for dual global/target continuity baseline evidence."""

from __future__ import annotations

import copy

import pytest

from analysis.physiology_synergy.calibrate_continuity_reward import (
    BASELINE_ROLLOUT_SCHEMA_VERSION,
    baseline_rollout_fingerprint,
    build_baseline_rollout_evidence,
    validate_baseline_rollout,
)
from tests.unit.continuity_v3_fixtures import (
    baseline_evidence,
    raw_samples,
    rollout_identity_and_manifest,
)


def _build(samples=None):
    identity, _manifest, _loss = rollout_identity_and_manifest()
    return build_baseline_rollout_evidence(
        identity,
        raw_samples() if samples is None else samples,
        expected_trajectory_count=2,
        expected_global_chain_count=28,
        expected_global_edge_count=140,
        expected_target_chain_count=4,
        expected_target_edge_count=20,
    )


def test_baseline_v3_seals_dual_losses_and_exact_target_identity():
    evidence = _build()

    assert evidence["schema_version"] == BASELINE_ROLLOUT_SCHEMA_VERSION
    assert evidence["coverage"] == {
        "trajectory_count": 2,
        "step_count": 64,
        "global_chain_count": 28,
        "global_edge_count": 140,
        "target_chain_count": 4,
        "target_edge_count": 20,
    }
    assert set(evidence["samples"]) == {
        "trajectory_index",
        "trajectory_step",
        "imitation_reward",
        "global_continuity_loss",
        "target_continuity_loss",
    }
    assert evidence["identity"]["candidate_loss_spec_fingerprint"]
    assert validate_baseline_rollout(evidence) == evidence


@pytest.mark.parametrize(
    ("field", "index", "value", "error"),
    [
        ("continuity_global_edge_count", 10, 139, "global-edge coverage"),
        ("continuity_target_chain_count", 10, 0, "target-chain coverage"),
        ("continuity_target_edge_count", 10, 19, "target-edge coverage"),
    ],
)
def test_baseline_v3_rejects_per_step_coverage_drift(field, index, value, error):
    samples = raw_samples()
    samples[field][index] = value
    with pytest.raises(ValueError, match=error):
        _build(samples)


def test_baseline_v3_rejects_padding_duplicates_and_noncontiguous_steps():
    duplicate = raw_samples()
    duplicate["trajectory_step"][-1] = 30
    with pytest.raises(ValueError, match="duplicate trajectory steps"):
        _build(duplicate)

    skipped = raw_samples()
    skipped["trajectory_step"][33] = 99
    with pytest.raises(ValueError, match="not one ordered prefix"):
        _build(skipped)


def test_baseline_v3_fingerprint_covers_target_samples_and_identity():
    baseline, _manifest, _identity = baseline_evidence()
    tampered = copy.deepcopy(baseline)
    tampered["samples"]["target_continuity_loss"][0] += 0.01
    with pytest.raises(ValueError, match="fingerprint is stale"):
        validate_baseline_rollout(tampered)

    tampered["artifact_fingerprint"] = baseline_rollout_fingerprint(tampered)
    assert validate_baseline_rollout(tampered) == tampered
