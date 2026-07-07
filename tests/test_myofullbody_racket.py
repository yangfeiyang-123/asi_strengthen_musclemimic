"""Tests for the rigid racket-holding MyoFullBody environment.

These run on CPU (jax backend put_model included) so they do not require a GPU.
"""

import mujoco
import numpy as np
import pytest

from musclemimic.environments.base import LocoEnv
from musclemimic.environments.humanoids.myofullbody import MyoFullBody
from musclemimic.environments.humanoids.myofullbody_racket import (
    RACKET_BODY_NAME,
    MjxMyoFullBodyRacket,
    MyoFullBodyRacket,
)


def _racket_geom_ids(model):
    return {
        i
        for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("racket_")
    }


def test_registered():
    assert "MyoFullBodyRacket" in LocoEnv.registered_envs
    assert "MjxMyoFullBodyRacket" in LocoEnv.registered_envs


def test_dims_match_plain_myofullbody():
    """The rigid racket must add zero DOF and zero actuators so retargeted
    free-hand trajectories and trained body policies transfer unchanged."""
    base = MyoFullBody(disable_fingers=True)._model
    racket = MyoFullBodyRacket(disable_fingers=True)._model
    assert (racket.nq, racket.nv, racket.nu) == (base.nq, base.nv, base.nu)


def test_racket_body_attached_to_hand():
    env = MyoFullBodyRacket(disable_fingers=True)
    m = env._model
    rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, RACKET_BODY_NAME)
    assert rid >= 0
    parent = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.body_parentid[rid]))
    assert parent == "thirdmc_r"
    assert float(m.body_mass[rid]) == pytest.approx(0.09, abs=1e-3)


def test_reset_step_finite():
    env = MyoFullBodyRacket(disable_fingers=True)
    obs = env.reset()
    assert np.all(np.isfinite(np.asarray(obs)))
    action = np.zeros(env._model.nu)
    for _ in range(5):
        env.step(action)


def test_racket_does_not_contact_body():
    env = MyoFullBodyRacket(disable_fingers=True)
    m = env._model
    d = mujoco.MjData(m)
    d.qpos[:] = env._data.qpos
    mujoco.mj_forward(m, d)
    racket = _racket_geom_ids(m)
    cross = sum(
        1
        for c in range(d.ncon)
        if (d.contact.geom1[c] in racket) ^ (d.contact.geom2[c] in racket)
    )
    assert cross == 0


def test_disable_racket_matches_plain():
    env = MyoFullBodyRacket(disable_fingers=True, enable_racket=False)
    m = env._model
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, RACKET_BODY_NAME) < 0


def test_mjx_put_model():
    from mujoco import mjx

    env = MjxMyoFullBodyRacket(mjx_backend="jax", disable_fingers=True, num_envs=2)
    mjx.put_model(env._model)  # raises on incompatibility


def test_retarget_alias():
    from loco_mujoco.smpl.retargeting import _resolve_retarget_env_name

    assert _resolve_retarget_env_name("MjxMyoFullBodyRacket") == "MjxMyoFullBody"
    assert _resolve_retarget_env_name("MyoFullBodyRacket") == "MyoFullBody"
    assert _resolve_retarget_env_name("MjxMyoFullBody") == "MjxMyoFullBody"
