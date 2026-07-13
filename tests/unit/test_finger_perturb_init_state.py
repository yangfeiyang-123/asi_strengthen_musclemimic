from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
from flax import struct

from loco_mujoco.core.initial_state_handler.finger_perturb_init_state import (
    FingerPerturbInitialStateHandler,
)
from loco_mujoco.core.initial_state_handler.traj_init_state import TrajInitialStateHandler


MODEL_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body>
      <joint name="wrist_flexion_r" type="hinge" range="-1 1"/>
      <joint name="cmc_flexion_r" type="hinge" range="-0.5 0.5"/>
      <joint name="mcp2_flexion_r" type="hinge" range="-0.4 0.4"/>
      <joint name="cmc_flexion_l" type="hinge" range="-0.5 0.5"/>
      <geom type="sphere" size="0.01" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _joint_state(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> tuple[float, float]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return (
        float(data.qpos[model.jnt_qposadr[joint_id]]),
        float(data.qvel[model.jnt_dofadr[joint_id]]),
    )


def test_right_hand_only_reset_perturbs_qpos_and_qvel_without_touching_left_or_wrist(monkeypatch):
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    data.qpos[:] = np.array([0.21, 0.10, -0.10, 0.31])
    data.qvel[:] = np.array([0.41, 0.20, -0.20, 0.51])
    baseline_qpos = data.qpos.copy()
    baseline_qvel = data.qvel.copy()
    env = SimpleNamespace(_model=model)

    # Isolate the subclass behavior: the base reset has already restored the
    # trajectory in production.
    monkeypatch.setattr(TrajInitialStateHandler, "reset", lambda self, *args, **kwargs: (data, None))
    handler = FingerPerturbInitialStateHandler(
        env,
        finger_perturb_side="right",
        finger_qpos_perturb_scale=0.05,
        finger_qvel_perturb_scale=0.10,
        finger_perturb_seed=123,
    )

    handler.reset(env, model, data, None, np)

    wrist_after = _joint_state(model, data, "wrist_flexion_r")
    left_after = _joint_state(model, data, "cmc_flexion_l")
    assert wrist_after == (baseline_qpos[0], baseline_qvel[0])
    assert left_after == (baseline_qpos[3], baseline_qvel[3])
    for name in ("cmc_flexion_r", "mcp2_flexion_r"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr = model.jnt_qposadr[joint_id]
        qvel_adr = model.jnt_dofadr[joint_id]
        assert abs(data.qpos[qpos_adr] - baseline_qpos[qpos_adr]) <= 0.05
        assert abs(data.qvel[qvel_adr] - baseline_qvel[qvel_adr]) <= 0.10
        assert data.qpos[qpos_adr] != baseline_qpos[qpos_adr]
        assert data.qvel[qvel_adr] != baseline_qvel[qvel_adr]


def test_legacy_finger_perturb_scale_defaults_to_both_hands_qpos_only(monkeypatch):
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    data.qvel[:] = np.array([0.1, 0.2, 0.3, 0.4])
    baseline_qvel = data.qvel.copy()
    env = SimpleNamespace(_model=model)
    monkeypatch.setattr(TrajInitialStateHandler, "reset", lambda self, *args, **kwargs: (data, None))

    handler = FingerPerturbInitialStateHandler(env, finger_perturb_scale=0.05, finger_perturb_seed=7)
    handler.reset(env, model, data, None, np)

    assert handler._finger_perturb_side == "both"
    assert handler._finger_qpos_perturb_scale == 0.05
    assert handler._finger_qvel_perturb_scale == 0.0
    np.testing.assert_array_equal(data.qvel, baseline_qvel)
    assert _joint_state(model, data, "cmc_flexion_r")[0] != 0.0
    assert _joint_state(model, data, "cmc_flexion_l")[0] != 0.0


def test_fold_in_rng_keeps_main_jax_key_and_changes_only_selected_fingers(monkeypatch):
    import jax
    import jax.numpy as jnp

    @struct.dataclass
    class FakeData:
        qpos: object
        qvel: object

    @struct.dataclass
    class FakeCarry:
        key: object

    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    base_qpos = jnp.asarray([0.21, 0.10, -0.10, 0.31])
    base_qvel = jnp.asarray([0.41, 0.20, -0.20, 0.51])
    env = SimpleNamespace(_model=model)
    monkeypatch.setattr(
        TrajInitialStateHandler,
        "reset",
        lambda self, *args, **kwargs: (FakeData(base_qpos, base_qvel), args[3]),
    )
    carry = FakeCarry(jax.random.PRNGKey(19))
    handler = FingerPerturbInitialStateHandler(
        env,
        finger_perturb_side="right",
        finger_qpos_perturb_scale=0.05,
        finger_qvel_perturb_scale=0.0,
        finger_perturb_rng_mode="fold_in",
    )

    perturbed, next_carry = handler.reset(env, model, FakeData(base_qpos, base_qvel), carry, jnp)

    np.testing.assert_array_equal(next_carry.key, carry.key)
    np.testing.assert_allclose(np.asarray(perturbed.qpos)[[0, 3]], np.asarray(base_qpos)[[0, 3]])
    assert not np.array_equal(np.asarray(perturbed.qpos)[1:3], np.asarray(base_qpos)[1:3])
    np.testing.assert_array_equal(perturbed.qvel, base_qvel)
