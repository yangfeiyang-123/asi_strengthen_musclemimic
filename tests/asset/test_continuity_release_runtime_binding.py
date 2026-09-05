"""Prepared-asset runtime checks for the immutable continuity release."""

from __future__ import annotations

import os

import pytest
from omegaconf import OmegaConf

from musclemimic.environments.humanoids.myofullbody import MyoFullBody
from musclemimic.environments.humanoids.myofullbody_racket import MyoFullBodyRacket
from musclemimic.physiology.anatomical_groups import validate_taxonomy_against_model
from musclemimic.physiology.release import (
    load_continuity_training_release,
    resolve_continuity_training_release,
    validate_release_against_runtime,
)
from musclemimic.runner.engine import bind_continuity_training_release

pytestmark = pytest.mark.asset


def _released_assets():
    path = os.environ.get("MUSCLEMIMIC_CONTINUITY_RELEASE", "").strip()
    if not path:
        pytest.skip("MUSCLEMIMIC_CONTINUITY_RELEASE is required for release runtime binding")
    release = load_continuity_training_release(path)
    return release, resolve_continuity_training_release(release), path


@pytest.mark.parametrize("action_mode", ["full_354", "fixed_synergy", "fixed_synergy_residual"])
def test_release_binds_every_declared_runtime_action_mode(tmp_path, action_mode):
    release, artifacts, release_path = _released_assets()
    assert action_mode in release.allowed_action_modes
    config = OmegaConf.create(
        {
            "experiment": {
                "action_representation": {
                    "enabled": action_mode != "full_354",
                    "mode": action_mode,
                },
                "continuity_ablation": {"action_mode": action_mode},
                "env_params": {
                    "reward_params": {
                        "intra_muscle_consistency": {
                            "mode": "reward",
                            "release_path": release_path,
                            "expected_release_fingerprint": release.release_fingerprint,
                        }
                    }
                },
            }
        }
    )
    contract = bind_continuity_training_release(
        config,
        launch_dir=tmp_path,
        result_dir=tmp_path / action_mode,
    )
    assert contract["release_fingerprint"] == release.release_fingerprint
    assert contract["loss_spec_fingerprint"] == artifacts.loss_identity.loss_spec_fingerprint
    assert contract["action_mode"] == action_mode


def test_release_keeps_one_portable_core_across_bare_and_racket_models():
    release, artifacts, _release_path = _released_assets()
    expected_core = release.taxonomy["muscle_channel_core_fingerprint"]
    for environment in (MyoFullBody(disable_fingers=True), MyoFullBodyRacket(disable_fingers=True)):
        validate_taxonomy_against_model(
            artifacts.taxonomy,
            environment._model,
            compatibility="portable_muscle_channel_abi",
        )
        assert artifacts.taxonomy.stable_model_binding["muscle_channel_core_fingerprint"] == expected_core
    for action_mode in release.allowed_action_modes:
        validate_release_against_runtime(
            release,
            taxonomy=artifacts.taxonomy,
            graph=artifacts.candidate_graph,
            runtime_loss_identity=artifacts.loss_identity,
            action_mode=action_mode,
        )
