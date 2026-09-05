"""A1 direct-354 continuity PPO GPU smoke evidence."""

from __future__ import annotations

import pytest

from tests.gpu._continuity_smoke import artifact_from_env

pytestmark = pytest.mark.gpu


def test_a1_full_354_reward_smoke_passed():
    artifact = artifact_from_env("MUSCLEMIMIC_A1_SMOKE_ARTIFACT")
    assert artifact["formal_config"]["condition"] == "A1"
    assert artifact["contracts"]["action_mode"] == "full_354"
    assert artifact["contracts"]["basis_family"] == "direct_354"
    assert artifact["contracts"]["basis_fingerprint"] is None
    assert artifact["checks"]["target_loss_nonconstant"] is True
    assert artifact["checks"]["checkpoint_restored"] is True
