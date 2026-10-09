"""B1/C1 fixed-synergy continuity PPO GPU smoke evidence."""

from __future__ import annotations

import pytest

from tests.gpu._continuity_smoke import artifact_from_env

pytestmark = pytest.mark.gpu


def test_b1_raw_unit_standard_nmf_smoke_passed():
    artifact = artifact_from_env("MUSCLEMIMIC_B1_SMOKE_ARTIFACT")
    assert artifact["formal_config"]["condition"] == "B1"
    assert artifact["contracts"]["action_mode"] == "fixed_synergy"
    assert artifact["contracts"]["basis_family"] == "standard_nmf"
    assert artifact["contracts"]["basis_fingerprint"] is not None
    assert artifact["contracts"]["basis_factor_contract_fingerprint"] is not None


def test_c1_structured_residual_smoke_passed():
    artifact = artifact_from_env("MUSCLEMIMIC_C1_SMOKE_ARTIFACT")
    assert artifact["formal_config"]["condition"] == "C1"
    assert artifact["contracts"]["action_mode"] == "fixed_synergy_residual"
    assert artifact["contracts"]["basis_family"] == "standard_nmf_structured_residual"
    assert artifact["contracts"]["basis_fingerprint"] is not None
    assert artifact["contracts"]["residual_basis_fingerprint"] is not None
