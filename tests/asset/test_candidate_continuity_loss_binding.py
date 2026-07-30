"""Prepared-asset binding of the reviewed candidate graph to MyoFullBody."""

from __future__ import annotations

import os

import pytest

from musclemimic.environments.humanoids.myofullbody import MyoFullBody
from musclemimic.physiology.anatomical_groups import validate_taxonomy_against_model
from musclemimic.physiology.continuity_groups import (
    assert_continuity_loss_spec_matches,
    build_continuity_loss_spec,
    validate_continuity_graph_against_model,
)
from musclemimic.physiology.release import (
    load_continuity_training_release,
    resolve_continuity_training_release,
)

pytestmark = pytest.mark.asset


def _released_assets():
    path = os.environ.get("MUSCLEMIMIC_CONTINUITY_RELEASE", "").strip()
    if not path:
        pytest.skip("MUSCLEMIMIC_CONTINUITY_RELEASE is required for reviewed candidate asset binding")
    release = load_continuity_training_release(path)
    return release, resolve_continuity_training_release(release)


def test_candidate_graph_compiles_the_exact_released_target_loss_on_myofullbody():
    release, artifacts = _released_assets()
    env = MyoFullBody(disable_fingers=True)
    validate_taxonomy_against_model(
        artifacts.taxonomy,
        env._model,
        compatibility="portable_muscle_channel_abi",
    )
    validate_continuity_graph_against_model(
        artifacts.candidate_graph,
        artifacts.taxonomy,
        env._model,
    )
    spec, runtime_identity = build_continuity_loss_spec(
        artifacts.candidate_graph,
        artifacts.taxonomy,
        training_enabled_only=True,
        signal=artifacts.loss_identity.signal,
        method=artifacts.loss_identity.method,
        scale=artifacts.loss_identity.scale,
        huber_delta=artifacts.loss_identity.huber_delta,
        eps=artifacts.loss_identity.eps,
    )
    assert_continuity_loss_spec_matches(artifacts.loss_identity, runtime_identity)
    assert runtime_identity.loss_spec_fingerprint == release.loss_spec["loss_spec_fingerprint"]
    assert runtime_identity.chain_count == release.loss_spec["target_chain_count"] > 0
    assert runtime_identity.edge_count == release.loss_spec["target_edge_count"] > 0
    assert tuple(spec.activation_addresses.shape) == (354,)
