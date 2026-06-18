from __future__ import annotations

import numpy as np
import mujoco

from environment.overall_environment.src import control_scaling
from environment.overall_environment.src.control_scaling import (
    apply_checkpoint_ctrl_ranges_to_model,
    normalized_action_to_model_ctrl,
)


def _model():
    xml = """
    <mujoco>
      <worldbody>
        <body name="body">
          <joint name="hinge0" type="hinge"/>
          <joint name="hinge1" type="hinge"/>
          <geom type="capsule" size="0.02 0.1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="positive" joint="hinge0" ctrllimited="true" ctrlrange="0 1"/>
        <motor name="signed" joint="hinge1" ctrllimited="true" ctrlrange="-2 2"/>
      </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def test_normalized_action_to_model_ctrl_matches_default_control_direct_scaling():
    model = _model()

    ctrl = normalized_action_to_model_ctrl(model, np.array([-1.0, 0.5]))

    np.testing.assert_allclose(ctrl, np.array([0.0, 1.0]), rtol=0.0, atol=0.0)


def test_normalized_action_to_model_ctrl_clips_after_scaling():
    model = _model()

    ctrl = normalized_action_to_model_ctrl(model, np.array([3.0, -3.0]))

    np.testing.assert_allclose(ctrl, np.array([1.0, -2.0]), rtol=0.0, atol=0.0)


def test_apply_checkpoint_ctrl_ranges_overrides_matching_actuators(monkeypatch):
    model = _model()

    monkeypatch.setattr(
        control_scaling,
        "checkpoint_actuator_ctrl_ranges",
        lambda _checkpoint: {"positive": (-1.0, 1.0), "missing": (-1.0, 1.0)},
    )
    report = apply_checkpoint_ctrl_ranges_to_model(model, "checkpoint")

    assert report.source_count == 2
    assert report.matched_count == 1
    assert report.changed_count == 1
    assert report.missing_actuators == ("missing",)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "positive")
    np.testing.assert_allclose(model.actuator_ctrlrange[actuator_id], np.array([-1.0, 1.0]))
