"""G1 graph-NMF continuity PPO GPU smoke evidence."""

from __future__ import annotations

import pytest

from tests.gpu._continuity_smoke import artifact_from_env

pytestmark = pytest.mark.gpu


def test_g1_graph_nmf_smoke_passed():
    artifact = artifact_from_env("MUSCLEMIMIC_G1_SMOKE_ARTIFACT")
    assert artifact["formal_config"]["condition"] == "G1"
    assert artifact["contracts"]["action_mode"] == "fixed_synergy"
    assert artifact["contracts"]["basis_family"] == "graph_nmf"
    assert artifact["contracts"]["basis_fingerprint"] is not None
    assert artifact["contracts"]["basis_factor_contract_fingerprint"] is not None
    assert artifact["contracts"]["graph_regularization_lineage_fingerprint"] is not None
    assert artifact["checks"]["graph_nmf_contract_matches"] is True
