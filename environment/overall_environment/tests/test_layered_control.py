from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.layered_control import LayeredActuatorRouter


def test_router_merges_body_and_grip_actions_by_name():
    router = LayeredActuatorRouter(
        all_actuator_names=["hip", "shoulder", "FDS2", "FDP2"],
        body_actuator_names=["hip", "shoulder"],
        grip_actuator_names=["FDS2", "FDP2"],
    )

    merged = router.merge(
        body_action=np.array([0.1, 0.2]),
        grip_action=np.array([0.7, 0.8]),
    )

    np.testing.assert_allclose(merged, np.array([0.1, 0.2, 0.7, 0.8]))
    assert router.source_labels() == ["body", "body", "grip", "grip"]


def test_router_rejects_overlapping_actuator_ownership():
    with pytest.raises(ValueError, match="owned by both"):
        LayeredActuatorRouter(
            all_actuator_names=["hip", "FDS2"],
            body_actuator_names=["hip", "FDS2"],
            grip_actuator_names=["FDS2"],
        )


def test_router_rejects_duplicate_body_actuator_names():
    with pytest.raises(ValueError, match="duplicate"):
        LayeredActuatorRouter(
            all_actuator_names=["hip", "shoulder", "FDS2"],
            body_actuator_names=["hip", "hip"],
            grip_actuator_names=["FDS2"],
        )


def test_router_rejects_duplicate_grip_actuator_names():
    with pytest.raises(ValueError, match="duplicate"):
        LayeredActuatorRouter(
            all_actuator_names=["hip", "FDS2", "FDP2"],
            body_actuator_names=["hip"],
            grip_actuator_names=["FDS2", "FDS2"],
        )


def test_router_validates_action_shapes():
    router = LayeredActuatorRouter(
        all_actuator_names=["hip", "FDS2"],
        body_actuator_names=["hip"],
        grip_actuator_names=["FDS2"],
    )

    with pytest.raises(ValueError, match="body_action"):
        router.merge(body_action=np.array([0.1, 0.2]), grip_action=np.array([0.3]))
