from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from musclemimic.distill.action_schema import actuator_schema_hash, ordered_schema_hash
from musclemimic.synergy.primitive_recording import write_primitive_trial_npz


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
<mujoco model="primitive-recording-fixture">
  <worldbody>
    <body name="body_a">
      <joint name="joint_a" type="hinge"/>
      <geom type="capsule" size="0.02 0.1" mass="1"/>
    </body>
    <body name="body_b" pos="0.2 0 0">
      <joint name="joint_b" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02 0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <general name="muscle_a" joint="joint_a" ctrllimited="true" ctrlrange="0 1"
      dyntype="muscle" gaintype="muscle" biastype="muscle"
      dynprm="0.01 0.04 0 0 0 0 0 0 0 0"
      gainprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0"
      biasprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0"
      lengthrange="0.1 1.0"/>
    <general name="muscle_b" joint="joint_b" ctrllimited="true" ctrlrange="0 1"
      dyntype="muscle" gaintype="muscle" biastype="muscle"
      dynprm="0.01 0.04 0 0 0 0 0 0 0 0"
      gainprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0"
      biasprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0"
      lengthrange="0.1 1.0"/>
  </actuator>
</mujoco>
"""
    )


def _ctrl() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [0.25, 0.25],
            [0.75, 0.75],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_writer_emits_ingest_ready_derived_contract_and_fingerprint(tmp_path):
    model = _model()
    path = tmp_path / "trial.npz"
    fingerprint = write_primitive_trial_npz(
        path,
        model=model,
        actuator_names=("muscle_a", "muscle_b"),
        applied_ctrl=_ctrl(),
        phase_id=np.asarray([0, 0, 1, 1], dtype=np.int32),
        success=True,
        muscle_activation=np.full((4, 2), 0.25),
        muscle_force=np.ones((4, 2)),
        muscle_tendon_length=np.ones((4, 2)),
        muscle_tendon_velocity=np.zeros((4, 2)),
        phase_local=np.asarray([0.0, 1.0, 0.0, 1.0]),
    )

    assert fingerprint == hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as trial:
        assert {
            "teacher_ctrl_physical",
            "muscle_excitation",
            "phase_id",
            "success",
            "actuator_names",
            "actuator_ctrlrange",
            "model_hash",
            "actuator_schema_hash",
            "ctrlrange_schema_hash",
            "transform_ctrlrange_schema_hash",
            "physical_signal_schema_version",
            "muscle_excitation_transform",
            "muscle_channel_contract_schema_version",
            "actuator_dyntype",
            "actuator_actnum",
            "actuator_actadr",
        }.issubset(trial.files)
        np.testing.assert_allclose(
            trial["muscle_excitation"],
            np.asarray(
                [[0.0, 0.0], [0.25, 0.25], [0.75, 0.75], [1.0, 1.0]],
                dtype=np.float32,
            ),
        )
        assert str(trial["model_hash"].tolist()) == hashlib.sha256(model.__getstate__()).hexdigest()
        assert str(trial["actuator_schema_hash"].tolist()) == actuator_schema_hash(("muscle_a", "muscle_b"))
        assert str(trial["ctrlrange_schema_hash"].tolist()) == ordered_schema_hash(
            kind="actuator_ctrlrange",
            payload={
                "actuator_names": ["muscle_a", "muscle_b"],
                "ctrlrange": [[0.0, 1.0], [0.0, 1.0]],
            },
        )


def test_writer_rejects_float_phase_and_wrong_complete_model_order(tmp_path):
    model = _model()
    with pytest.raises(ValueError, match="phase_id must be an integer"):
        write_primitive_trial_npz(
            tmp_path / "float-phase.npz",
            model=model,
            actuator_names=("muscle_a", "muscle_b"),
            teacher_ctrl_physical=_ctrl(),
            phase_id=np.asarray([0.0, 0.0, 1.0, 1.0]),
            success=True,
        )

    with pytest.raises(ValueError, match="complete actuator order"):
        write_primitive_trial_npz(
            tmp_path / "wrong-order.npz",
            model=model,
            actuator_names=("muscle_b", "muscle_a"),
            teacher_ctrl_physical=_ctrl(),
            phase_id=np.asarray([0, 0, 1, 1], dtype=np.int32),
            success=True,
        )


def test_writer_rejects_out_of_range_physical_control(tmp_path):
    ctrl = _ctrl()
    ctrl[0, 0] = -0.000001
    with pytest.raises(ValueError, match="outside the exact compiled model ctrlrange"):
        write_primitive_trial_npz(
            tmp_path / "out-of-range.npz",
            model=_model(),
            actuator_names=("muscle_a", "muscle_b"),
            teacher_ctrl_physical=ctrl,
            phase_id=np.asarray([0, 0, 1, 1], dtype=np.int32),
            success=True,
        )


def test_writer_refuses_overwrite_by_default(tmp_path):
    path = tmp_path / "trial.npz"
    kwargs = {
        "model": _model(),
        "actuator_names": ("muscle_a", "muscle_b"),
        "teacher_ctrl_physical": _ctrl(),
        "phase_id": np.asarray([0, 0, 1, 1], dtype=np.int32),
        "success": True,
    }
    original = write_primitive_trial_npz(path, **kwargs)

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        write_primitive_trial_npz(path, **kwargs)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
