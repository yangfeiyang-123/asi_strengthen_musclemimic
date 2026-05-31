from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.action_adapter import CheckpointToFullActionAdapter


def test_adapter_maps_by_name_not_index():
    adapter = CheckpointToFullActionAdapter(
        source_actuator_names=["shoulder", "hip"],
        target_actuator_names=["hip", "FDS2", "shoulder"],
    )

    full = adapter.adapt(np.array([0.2, 0.7]))

    np.testing.assert_allclose(full, np.array([0.7, 0.0, 0.2]))
    assert adapter.report().mapped_count == 2
    assert adapter.report().extra_in_target == ["FDS2"]


def test_adapter_rejects_missing_source_actuator_in_target():
    with pytest.raises(ValueError, match="source actuators missing in target"):
        CheckpointToFullActionAdapter(
            source_actuator_names=["hip", "missing"],
            target_actuator_names=["hip"],
        )


def test_adapter_sets_extra_target_actuators_to_zero():
    adapter = CheckpointToFullActionAdapter(["hip"], ["hip", "FDS2"])

    np.testing.assert_allclose(adapter.adapt(np.array([0.4])), np.array([0.4, 0.0]))


def test_adapter_rejects_nonfinite_action():
    adapter = CheckpointToFullActionAdapter(["hip"], ["hip"])

    with pytest.raises(ValueError, match="non-finite"):
        adapter.adapt(np.array([np.nan]))
