"""Calibration must use the exact candidate target loss, never the full graph."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from analysis.physiology_synergy.calibrate_continuity_reward import (
    CALIBRATION_SCHEMA_VERSION,
    PENALTY_BUDGET_FRACTIONS,
    baseline_rollout_fingerprint,
    build_continuity_reward_calibration,
    calibration_fingerprint,
    validate_continuity_reward_calibration,
)
from tests.unit.continuity_v3_fixtures import baseline_evidence


def _calibrate(baseline=None, manifest=None, loss_identity=None, *, fraction=0.01):
    fixture_baseline, fixture_manifest, fixture_identity = baseline_evidence()
    return build_continuity_reward_calibration(
        fixture_baseline if baseline is None else baseline,
        rollout_manifest=fixture_manifest if manifest is None else manifest,
        expected_loss_spec=fixture_identity if loss_identity is None else loss_identity,
        selected_budget_fraction=fraction,
    )


def test_calibration_v3_uses_target_quantile_and_binds_exact_loss_spec():
    baseline, _manifest, loss_identity = baseline_evidence()
    calibration = _calibrate(baseline=baseline, loss_identity=loss_identity)

    target_q95 = float(np.quantile(baseline["samples"]["target_continuity_loss"], 0.95))
    global_q95 = float(np.quantile(baseline["samples"]["global_continuity_loss"], 0.95))
    median_reward = float(np.quantile(baseline["samples"]["imitation_reward"], 0.5))
    assert calibration["schema_version"] == CALIBRATION_SCHEMA_VERSION
    assert calibration["statistics"]["q95_target_continuity_loss"] == pytest.approx(target_q95)
    assert target_q95 != pytest.approx(global_q95)
    assert calibration["selection"]["coefficient"] == pytest.approx(0.01 * median_reward / target_q95)
    assert calibration["target_loss_spec"] == {
        "candidate_graph_fingerprint": loss_identity.graph_fingerprint,
        "candidate_loss_spec_fingerprint": loss_identity.loss_spec_fingerprint,
        "target_chain_count": loss_identity.chain_count,
        "target_edge_count": loss_identity.edge_count,
    }
    assert [row["fraction_of_median_imitation_reward"] for row in calibration["candidate_budgets"]] == list(
        PENALTY_BUDGET_FRACTIONS
    )
    assert validate_continuity_reward_calibration(calibration) == calibration


def test_calibration_v3_rejects_full_only_and_all_zero_target_evidence():
    baseline, manifest, loss_identity = baseline_evidence()
    full_only = copy.deepcopy(baseline)
    del full_only["samples"]["target_continuity_loss"]
    full_only["artifact_fingerprint"] = baseline_rollout_fingerprint(full_only)
    with pytest.raises(ValueError, match="baseline samples fields differ"):
        _calibrate(baseline=full_only, manifest=manifest, loss_identity=loss_identity)

    zero, zero_manifest, zero_identity = baseline_evidence(target_all_zero=True)
    with pytest.raises(ValueError, match="target continuity loss cannot be all zero"):
        _calibrate(baseline=zero, manifest=zero_manifest, loss_identity=zero_identity)


def test_calibration_v3_rejects_loss_spec_graph_or_fingerprint_drift():
    baseline, manifest, loss_identity = baseline_evidence()
    drifted_manifest = loss_identity.to_manifest()
    drifted_manifest["loss_spec_fingerprint"] = "9" * 64
    with pytest.raises(ValueError, match=r"fingerprint is stale|differs from baseline"):
        _calibrate(
            baseline=baseline,
            manifest=manifest,
            loss_identity=drifted_manifest,
        )

    drifted_baseline = copy.deepcopy(baseline)
    drifted_baseline["identity"]["candidate_loss_spec_fingerprint"] = "8" * 64
    drifted_baseline["artifact_fingerprint"] = baseline_rollout_fingerprint(drifted_baseline)
    with pytest.raises(ValueError, match="identity differs"):
        _calibrate(
            baseline=drifted_baseline,
            manifest=manifest,
            loss_identity=loss_identity,
        )


def test_calibration_v3_fingerprint_covers_coefficient_and_target_binding():
    calibration = _calibrate(fraction=0.02)
    tampered = copy.deepcopy(calibration)
    tampered["selection"]["coefficient"] *= 2.0
    with pytest.raises(ValueError, match="selected coefficient is stale"):
        validate_continuity_reward_calibration(tampered)

    tampered = copy.deepcopy(calibration)
    tampered["target_loss_spec"]["target_edge_count"] -= 1
    tampered["calibration_fingerprint"] = calibration_fingerprint(tampered)
    with pytest.raises(ValueError, match="binding differs"):
        validate_continuity_reward_calibration(tampered)
