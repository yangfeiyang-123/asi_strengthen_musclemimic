"""Tests for MimicReward XY offset correction in qpos comparison."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from flax import struct

from musclemimic.core.reward.trajectory_based import (
    MimicReward,
    MimicRewardState,
    _ordered_muscle_activation_addresses,
)


@struct.dataclass
class FakeTrajState:
    """Trajectory state for testing."""

    traj_no: int = 0
    subtraj_step_no: int = 0
    subtraj_step_no_init: int = 0


@struct.dataclass
class FakeCarry:
    """Carry object for testing that supports .replace()."""

    traj_state: FakeTrajState
    reward_state: MimicRewardState
    qvel_w_sum: float = 0.1
    root_vel_w_sum: float = 0.1


class FakeCarryNoInit:
    """Carry without subtraj_step_no_init for testing offset skip case."""

    def __init__(self, traj_state, reward_state):
        self.traj_state = traj_state
        self.reward_state = reward_state
        self.qvel_w_sum = 0.1
        self.root_vel_w_sum = 0.1

    def replace(self, **kwargs):
        """Support .replace() for compatibility with real carry."""
        new_carry = FakeCarryNoInit(self.traj_state, self.reward_state)
        for key, value in kwargs.items():
            setattr(new_carry, key, value)
        return new_carry


# Minimal MJCF for testing with a free joint
MINIMAL_MJCF = """
<mujoco model="test_mimic_reward">
  <worldbody>
    <body name="root">
      <joint name="root_joint" type="free"/>
      <geom name="torso" size="0.1"/>
      <site name="pelvis_mimic" pos="0 0 0"/>
      <site name="upper_body_mimic" pos="0 0 0.1"/>
      <body name="child" pos="0 0 0.2">
        <joint name="hinge1" type="hinge" axis="1 0 0"/>
        <geom name="link1" size="0.05"/>
        <site name="child_mimic" pos="0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

MINIMAL_BIMANUAL_MJCF = """
<mujoco model="test_bimanual_mimic_reward">
  <worldbody>
    <body name="thorax">
      <joint name="shoulder_l" type="hinge" axis="1 0 0"/>
      <geom name="thorax_geom" size="0.1"/>
      <site name="upper_body_mimic" pos="0 0 0"/>
      <body name="forearm" pos="0 0 0.2">
        <joint name="elbow_l" type="hinge" axis="0 1 0"/>
        <geom name="link1" size="0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""



class FakeTrajData:
    """Minimal trajectory data for testing."""

    def __init__(self, qpos, qvel, site_xpos, site_xmat, xpos, xquat, cvel, subtree_com):
        self.qpos = qpos
        self.qvel = qvel
        self.site_xpos = site_xpos
        self.site_xmat = site_xmat
        self.xpos = xpos
        self.xquat = xquat
        self.cvel = cvel
        self.subtree_com = subtree_com


class FakeTrajInfo:
    """Trajectory info with site names."""

    def __init__(self, site_names):
        self.site_names = site_names


class FakeTraj:
    """Container for trajectory data and info."""

    def __init__(self, data, info):
        self.data = data
        self.info = info


class FakeTrajectoryHandler:
    """Mock trajectory handler for testing."""

    def __init__(self, current_data, init_data=None, traj_info=None):
        self._current_data = current_data
        self._init_data = init_data or current_data
        self._traj_info = traj_info or FakeTrajInfo(["pelvis_mimic", "upper_body_mimic", "child_mimic"])
        self.traj = FakeTraj(None, self._traj_info)

    def get_current_traj_data(self, carry, backend):
        return self._current_data

    def get_traj_data_at(self, traj_no, subtraj_step_no, carry, backend):
        # Return init data for init step, current for others
        if hasattr(carry, "traj_state") and subtraj_step_no == carry.traj_state.subtraj_step_no_init:
            return self._init_data
        return self._current_data


def make_traj_data(qpos, qvel=None, n_sites=3, n_bodies=3, backend=np):
    """Create fake trajectory data with given qpos.

    Note: MuJoCo models have world body (id=0) plus user bodies, so n_bodies=3
    for a model with root + child bodies.
    """
    nq = len(qpos)
    nv = nq - 1 if nq >= 7 else nq  # Free joint: 7 qpos, 6 qvel
    if qvel is None:
        qvel = backend.zeros(nv)
    return FakeTrajData(
        qpos=backend.asarray(qpos),
        qvel=backend.asarray(qvel),
        site_xpos=backend.zeros((n_sites, 3)),
        site_xmat=backend.tile(backend.eye(3).reshape(1, 9), (n_sites, 1)),
        xpos=backend.zeros((n_bodies, 3)),
        xquat=backend.tile(backend.array([1.0, 0.0, 0.0, 0.0]), (n_bodies, 1)),
        cvel=backend.zeros((n_bodies, 6)),
        subtree_com=backend.zeros((n_bodies, 3)),
    )


def make_carry(traj_no=0, subtraj_step_no=10, subtraj_step_no_init=0, include_init=True):
    """Create fake carry with trajectory state."""
    reward_state = MimicRewardState(last_qvel=np.zeros(7), last_action=np.zeros(3), imitation_error_total=0.0)
    if include_init:
        traj_state = FakeTrajState(
            traj_no=traj_no, subtraj_step_no=subtraj_step_no, subtraj_step_no_init=subtraj_step_no_init
        )
        return FakeCarry(traj_state=traj_state, reward_state=reward_state)
    else:
        # For no-init case, use SimpleNamespace without subtraj_step_no_init attribute
        traj_state = SimpleNamespace(traj_no=traj_no, subtraj_step_no=subtraj_step_no)
        # Create a carry-like object that supports .replace() via a wrapper
        return FakeCarryNoInit(traj_state=traj_state, reward_state=reward_state)


def make_env(
    model,
    th,
    *,
    root_free_joint_xml_name="root_joint",
    upper_body_xml_name="root",
    sites_for_mimic=None,
):
    """Create mock environment for testing."""
    if sites_for_mimic is None:
        sites_for_mimic = ["pelvis_mimic", "upper_body_mimic", "child_mimic"]
    env = MagicMock()
    env._model = model
    env.th = th
    env.dt = 0.01
    env.mdp_info = MagicMock()
    env.mdp_info.action_space = MagicMock()
    env.mdp_info.action_space.low = np.array([-1, -1, -1])
    env.mdp_info.action_space.high = np.array([1, 1, 1])
    env._get_all_info_properties = lambda: {
        "root_free_joint_xml_name": root_free_joint_xml_name,
        "upper_body_xml_name": upper_body_xml_name,
        "sites_for_mimic": sites_for_mimic,
    }
    env.sites_for_mimic = sites_for_mimic
    env.info = MagicMock()
    env.info.action_space = MagicMock()
    env.info.action_space.shape = (3,)
    return env


def make_sim_data(qpos, qvel=None, n_sites=3, n_bodies=3, backend=np):
    """Create simulation data (MjData-like) with given qpos.

    Note: MuJoCo models have world body (id=0) plus user bodies, so n_bodies=3
    for a model with root + child bodies.
    """
    nq = len(qpos)
    nv = nq - 1 if nq >= 7 else nq
    if qvel is None:
        qvel = backend.zeros(nv)
    return SimpleNamespace(
        qpos=backend.asarray(qpos),
        qvel=backend.asarray(qvel),
        site_xpos=backend.zeros((n_sites, 3)),
        site_xmat=backend.tile(backend.eye(3).reshape(1, 9), (n_sites, 1)),
        xpos=backend.zeros((n_bodies, 3)),
        xquat=backend.tile(backend.array([1.0, 0.0, 0.0, 0.0]), (n_bodies, 1)),
        cvel=backend.zeros((n_bodies, 6)),
        subtree_com=backend.zeros((n_bodies, 3)),
        qfrc_actuator=backend.zeros(nv),
        act=backend.zeros(3),
    )


# =====================================================================
# Tests for XY offset correction initialization
# =====================================================================


def test_mimic_reward_stores_root_xy_indices():
    """MimicReward should store root XY qpos indices for offset correction."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    traj_data = make_traj_data([0.0] * 8)  # 7 qpos (free) + 1 hinge
    th = FakeTrajectoryHandler(traj_data)
    env = make_env(model, th)

    reward = MimicReward(env)

    # Should have stored root XY indices (0, 1)
    assert reward._root_qpos_ids_xy is not None
    np.testing.assert_array_equal(reward._root_qpos_ids_xy, [0, 1])


def test_mimic_reward_stores_root_xy_in_qpos_ind():
    """MimicReward should map root XY to positions within _qpos_ind."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    traj_data = make_traj_data([0.0] * 8)
    th = FakeTrajectoryHandler(traj_data)
    env = make_env(model, th)

    reward = MimicReward(env)

    # _root_xy_in_qpos_ind should point to positions 0, 1 within _qpos_ind
    # since the free joint is first
    assert reward._root_xy_in_qpos_ind is not None
    np.testing.assert_array_equal(reward._root_xy_in_qpos_ind, [0, 1])


# =====================================================================
# Tests for XY offset correction in reward calculation
# =====================================================================


def test_qpos_xy_offset_correction_numpy():
    """Qpos comparison should apply XY offset correction with numpy backend."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)

    # Init qpos at (10, 5, z) in world coords
    init_qpos = np.array([10.0, 5.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # 7 free + 1 hinge
    # Current qpos in world coords at (10.5, 5.2, z)
    current_qpos = np.array([10.5, 5.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    init_data = make_traj_data(init_qpos, backend=np)
    current_data = make_traj_data(current_qpos, backend=np)
    th = FakeTrajectoryHandler(current_data, init_data)
    env = make_env(model, th)

    reward = MimicReward(env)
    carry = make_carry(subtraj_step_no=10, subtraj_step_no_init=0)

    # Simulation qpos is offset-adjusted: should be at (0.5, 0.2, z) after init offset
    sim_qpos = np.array([0.5, 0.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    sim_data = make_sim_data(sim_qpos, backend=np)

    # Call reward - should apply offset correction internally
    _, _, reward_info = reward(
        state=np.zeros(10),
        action=np.zeros(3),
        next_state=np.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=sim_data,
        carry=carry,
        backend=np,
    )

    # If offset correction works, sim_qpos (0.5, 0.2) should match
    # corrected traj_qpos (10.5-10, 5.2-5) = (0.5, 0.2)
    np.testing.assert_allclose(reward_info["reward_qpos"], 1.0, atol=1e-5)


def test_qpos_xy_offset_correction_jax():
    """Qpos comparison should apply XY offset correction with JAX backend."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)

    # Init qpos at (20, 10, z) in world coords
    init_qpos = jnp.array([20.0, 10.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    # Current qpos in world coords at (21.0, 11.0, z)
    current_qpos = jnp.array([21.0, 11.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    init_data = make_traj_data(init_qpos, backend=jnp)
    current_data = make_traj_data(current_qpos, backend=jnp)
    th = FakeTrajectoryHandler(current_data, init_data)
    env = make_env(model, th)

    reward = MimicReward(env)
    carry = make_carry(subtraj_step_no=10, subtraj_step_no_init=0)

    # Simulation qpos is offset-adjusted: (1.0, 1.0, z) after init offset
    sim_qpos = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    sim_data = make_sim_data(sim_qpos, backend=jnp)

    _, _, reward_info = reward(
        state=jnp.zeros(10),
        action=jnp.zeros(3),
        next_state=jnp.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=sim_data,
        carry=carry,
        backend=jnp,
    )

    # Corrected traj_qpos should be (21-20, 11-10) = (1, 1), matching sim_qpos
    np.testing.assert_allclose(reward_info["reward_qpos"], 1.0, atol=1e-5)


def test_qpos_xy_offset_correction_random_start_init():
    """XY offset correction should use the random-start init step, not origin."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)

    # Random start (non-zero) for trajectory init step
    init_qpos = np.array([2.5, -1.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.25])
    current_qpos = np.array([2.8, -0.4, 1.0, 1.0, 0.0, 0.0, 0.0, 0.25])

    init_data = make_traj_data(init_qpos, backend=np)
    current_data = make_traj_data(current_qpos, backend=np)
    th = FakeTrajectoryHandler(current_data, init_data)
    original_get_traj_data_at = th.get_traj_data_at
    th.get_traj_data_at = MagicMock(side_effect=original_get_traj_data_at)
    env = make_env(model, th)

    reward = MimicReward(env)
    carry = make_carry(subtraj_step_no=12, subtraj_step_no_init=7)

    # Simulation qpos should match trajectory after init offset: (0.3, 1.1, ...)
    sim_qpos = np.array([0.3, 1.1, 1.0, 1.0, 0.0, 0.0, 0.0, 0.25])
    sim_data = make_sim_data(sim_qpos, backend=np)

    _, _, reward_info = reward(
        state=np.zeros(10),
        action=np.zeros(3),
        next_state=np.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=sim_data,
        carry=carry,
        backend=np,
    )

    np.testing.assert_allclose(reward_info["reward_qpos"], 1.0, atol=1e-5)
    assert th.get_traj_data_at.call_args[0][1] == carry.traj_state.subtraj_step_no_init


def test_diagnostic_errors_apply_xy_offset():
    """Diagnostic errors should compare in local frame after XY offset."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)

    init_qpos = np.array([10.0, 5.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    current_qpos = np.array([10.5, 5.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    init_data = make_traj_data(init_qpos, backend=np)
    current_data = make_traj_data(current_qpos, backend=np)
    traj_sites = np.array(
        [
            [10.1, 5.1, 0.0],
            [10.6, 5.3, 0.0],
            [11.0, 5.4, 0.1],
        ]
    )
    current_data.site_xpos = traj_sites

    th = FakeTrajectoryHandler(current_data, init_data)
    env = make_env(model, th)
    reward = MimicReward(env)
    carry = make_carry(subtraj_step_no=10, subtraj_step_no_init=0)

    sim_qpos = np.array([0.5, 0.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    sim_data = make_sim_data(sim_qpos, backend=np)
    sim_data.site_xpos = traj_sites - np.array([10.0, 5.0, 0.0])

    _, _, reward_info = reward(
        state=np.zeros(10),
        action=np.zeros(3),
        next_state=np.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=sim_data,
        carry=carry,
        backend=np,
    )

    np.testing.assert_allclose(reward_info["err_root_xyz"], 0.0, atol=1e-6)
    np.testing.assert_allclose(reward_info["err_site_abs"], 0.0, atol=1e-6)


def test_bimanual_joint_diagnostic_errors_without_free_joint():
    """Fixed-root bimanual models should still report joint position and velocity errors."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_BIMANUAL_MJCF)

    traj_qpos = np.array([0.2, -0.1])
    traj_qvel = np.array([0.1, -0.2])
    current_data = make_traj_data(traj_qpos, qvel=traj_qvel, n_sites=1, backend=np)
    th = FakeTrajectoryHandler(current_data, traj_info=FakeTrajInfo(["upper_body_mimic"]))
    env = make_env(
        model,
        th,
        root_free_joint_xml_name="none",
        upper_body_xml_name="thorax",
        sites_for_mimic=["upper_body_mimic"],
    )

    reward = MimicReward(
        env,
        qpos_w_sum=0.0,
        qvel_w_sum=0.0,
        root_pos_w_sum=0.0,
        root_vel_w_sum=0.0,
        rpos_w_sum=0.0,
        rquat_w_sum=0.0,
        rvel_w_sum=0.0,
    )
    carry = make_carry()
    carry = carry.replace(
        reward_state=MimicRewardState(last_qvel=np.zeros(model.nv), last_action=np.zeros(3), imitation_error_total=0.0),
        qvel_w_sum=0.0,
        root_vel_w_sum=0.0,
    )

    sim_qpos = np.array([0.5, 0.3])
    sim_qvel = np.array([0.4, 0.2])
    sim_data = make_sim_data(sim_qpos, qvel=sim_qvel, n_sites=1, backend=np)

    _, _, reward_info = reward(
        state=np.zeros(2),
        action=np.zeros(3),
        next_state=np.zeros(2),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=sim_data,
        carry=carry,
        backend=np,
    )

    expected = np.sqrt(0.125)
    np.testing.assert_allclose(reward_info["err_joint_pos"], expected, atol=1e-6)
    np.testing.assert_allclose(reward_info["err_joint_vel"], expected, atol=1e-6)
    np.testing.assert_allclose(reward_info["err_root_xyz"], 0.0, atol=1e-6)
    np.testing.assert_allclose(reward_info["err_root_yaw"], 0.0, atol=1e-6)


def test_qpos_xy_offset_not_applied_without_init():
    """Offset correction should be skipped when subtraj_step_no_init is missing."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)

    # Current qpos at world position (10, 5, z)
    current_qpos = np.array([10.0, 5.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    current_data = make_traj_data(current_qpos, backend=np)
    th = FakeTrajectoryHandler(current_data)
    env = make_env(model, th)

    reward = MimicReward(env)
    # Carry without subtraj_step_no_init
    carry = make_carry(subtraj_step_no=10, include_init=False)

    # Simulation qpos is near origin (offset-adjusted)
    sim_qpos = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    sim_data = make_sim_data(sim_qpos, backend=np)

    _, _, reward_info = reward(
        state=np.zeros(10),
        action=np.zeros(3),
        next_state=np.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=sim_data,
        carry=carry,
        backend=np,
    )

    # Without offset correction, there's a mismatch: sim (0, 0) vs traj (10, 5)
    assert reward_info["reward_qpos"] < 1e-3


def test_unweighted_saturation_and_activation_diagnostics_are_always_reported():
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    qpos = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    th = FakeTrajectoryHandler(make_traj_data(qpos, backend=np))
    env = make_env(model, th)
    reward = MimicReward(
        env,
        action_rate_coeff=0.0,
        activation_energy_coeff=0.0,
    )
    # Exercise the actadr-indexed contract independently of this minimal
    # actuator-free MJCF.  Non-muscle activation states must be ignored.
    reward._muscle_activation_addresses = np.asarray([1], dtype=np.int32)
    carry = make_carry()
    sim_data = make_sim_data(qpos, backend=np)
    sim_data.act = np.asarray([9.0, 0.5, 8.0])

    _, _, reward_info = reward(
        state=np.zeros(10),
        action=np.asarray([0.99, 0.0, -1.0]),
        next_state=np.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=sim_data,
        carry=carry,
        backend=np,
    )

    assert reward_info["penalty_activation_energy"] == 0.0
    assert reward_info["activation_energy"] == pytest.approx(0.25)
    assert reward_info["action_saturation_fraction"] == pytest.approx(2.0 / 3.0)
    assert reward_info["action_rate_mean_square"] > 0.0


def test_muscle_activation_addresses_use_actadr_not_actuator_id():
    model = SimpleNamespace(
        actuator_dyntype=np.asarray(
            [
                int(mujoco.mjtDyn.mjDYN_NONE),
                int(mujoco.mjtDyn.mjDYN_MUSCLE),
                int(mujoco.mjtDyn.mjDYN_FILTER),
                int(mujoco.mjtDyn.mjDYN_MUSCLE),
            ]
        ),
        actuator_actnum=np.asarray([0, 1, 1, 1]),
        actuator_actadr=np.asarray([-1, 2, 0, 1]),
        na=3,
    )

    np.testing.assert_array_equal(
        _ordered_muscle_activation_addresses(model),
        np.asarray([2, 1], dtype=np.int32),
    )


def test_action_saturation_penalty_uses_diagnostic_boundary_band():
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    qpos = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    th = FakeTrajectoryHandler(make_traj_data(qpos, backend=np))
    env = make_env(model, th)
    reward = MimicReward(
        env,
        action_saturation_coeff=0.01,
        action_saturation_margin_fraction=0.02,
    )
    carry = make_carry()
    sim_data = make_sim_data(qpos, backend=np)

    _, _, reward_info = reward(
        state=np.zeros(10),
        action=np.asarray([0.99, 0.0, -1.0]),
        next_state=np.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=sim_data,
        carry=carry,
        backend=np,
    )

    # For [-1, 1] actions the 2% boundary band is 0.04 wide.  The normalized
    # violations are 0.75, 0.0 and 1.0 for the three actions above.
    expected_cost = (0.75**2 + 0.0 + 1.0**2) / 3.0
    assert reward_info["action_saturation_fraction"] == pytest.approx(2.0 / 3.0)
    assert reward_info["penalty_action_saturation"] == pytest.approx(
        -0.01 * expected_cost
    )

    # Raw Gaussian samples can overshoot far outside the action space.  The
    # saturation component is bounded per dimension; out-of-bounds magnitude
    # is handled separately and must not force this component below -coeff.
    _, _, overshoot_info = reward(
        state=np.zeros(10),
        action=np.asarray([3.0, -3.0, 0.0]),
        next_state=np.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=sim_data,
        carry=carry,
        backend=np,
    )
    assert overshoot_info["penalty_action_saturation"] == pytest.approx(
        -0.01 * (2.0 / 3.0)
    )


def test_action_saturation_margin_fraction_is_validated():
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    qpos = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    th = FakeTrajectoryHandler(make_traj_data(qpos, backend=np))
    env = make_env(model, th)

    with pytest.raises(ValueError, match="action_saturation_margin_fraction"):
        MimicReward(env, action_saturation_margin_fraction=0.5)


def test_qpos_xy_offset_exact_match():
    """When sim and corrected traj qpos match exactly, qpos reward should be ~1."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)

    # Set up exact match scenario
    init_qpos = np.array([100.0, 200.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5])
    current_qpos = np.array([100.0, 200.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5])  # Same as init

    init_data = make_traj_data(init_qpos, backend=np)
    current_data = make_traj_data(current_qpos, backend=np)
    th = FakeTrajectoryHandler(current_data, init_data)
    env = make_env(model, th)

    # Force qpos reward weight to be high
    reward = MimicReward(env, qpos_w_sum=1.0, qvel_w_sum=0.0, rpos_w_sum=0.0, rquat_w_sum=0.0, rvel_w_sum=0.0)
    carry = make_carry(subtraj_step_no=0, subtraj_step_no_init=0)

    # Sim qpos after offset: (0, 0, z, quat, hinge)
    # Corrected traj qpos: (100-100, 200-200, z, quat, hinge) = (0, 0, z, quat, hinge)
    sim_qpos = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5])
    sim_data = make_sim_data(sim_qpos, backend=np)

    total_reward, _, reward_info = reward(
        state=np.zeros(10),
        action=np.zeros(3),
        next_state=np.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=sim_data,
        carry=carry,
        backend=np,
    )

    # qpos reward should be close to 1.0 when all qpos match
    np.testing.assert_allclose(reward_info["reward_qpos"], 1.0, atol=1e-5)


def test_root_position_reward_uses_offset_corrected_root_xyz():
    """Explicit root position reward should match in local frame after XY offset correction."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)

    init_qpos = np.array([10.0, 5.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    current_qpos = np.array([10.5, 5.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    init_data = make_traj_data(init_qpos, backend=np)
    current_data = make_traj_data(current_qpos, backend=np)
    th = FakeTrajectoryHandler(current_data, init_data)
    env = make_env(model, th)

    reward = MimicReward(
        env,
        qpos_w_sum=0.0,
        qvel_w_sum=0.0,
        root_pos_w_sum=1.0,
        root_vel_w_sum=0.0,
        rpos_w_sum=0.0,
        rquat_w_sum=0.0,
        rvel_w_sum=0.0,
    )
    carry = make_carry(subtraj_step_no=10, subtraj_step_no_init=0)
    carry = carry.replace(qvel_w_sum=0.0, root_vel_w_sum=0.0)

    sim_qpos = np.array([0.5, 0.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    sim_data = make_sim_data(sim_qpos, backend=np)

    total_reward, _, reward_info = reward(
        state=np.zeros(10),
        action=np.zeros(3),
        next_state=np.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=sim_data,
        carry=carry,
        backend=np,
    )

    np.testing.assert_allclose(reward_info["reward_root_pos"], 1.0, atol=1e-5)
    np.testing.assert_allclose(total_reward, 1.0, atol=1e-5)
    np.testing.assert_allclose(reward_info["err_root_xyz"], 0.0, atol=1e-6)


def test_dedicated_root_rotation_and_angular_velocity_rewards():
    """Global root errors must have direct rewards and unweighted diagnostics."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    ref_qpos = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    ref_qvel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    ref_data = make_traj_data(ref_qpos, qvel=ref_qvel, backend=np)
    env = make_env(model, FakeTrajectoryHandler(ref_data))

    reward = MimicReward(
        env,
        qpos_w_sum=0.0,
        qvel_w_sum=0.0,
        root_pos_w_sum=0.0,
        root_vel_w_sum=0.0,
        rpos_w_sum=0.0,
        rquat_w_sum=0.0,
        rvel_w_sum=0.0,
        root_orientation_w_sum=0.5,
        root_orientation_w_exp=8.0,
        root_ang_vel_w_sum=0.5,
        root_ang_vel_w_exp=0.5,
    )
    carry = make_carry().replace(qvel_w_sum=0.0, root_vel_w_sum=0.0)

    angle = 0.5
    sim_qpos = ref_qpos.copy()
    sim_qpos[3:7] = [np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)]
    sim_qvel = ref_qvel.copy()
    sim_qvel[5] = 4.0
    sim_data = make_sim_data(sim_qpos, qvel=sim_qvel, backend=np)

    total_reward, _, info = reward(
        state=np.zeros(10),
        action=np.zeros(3),
        next_state=np.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=sim_data,
        carry=carry,
        backend=np,
    )

    expected_rot_reward = np.exp(-8.0 * angle**2)
    expected_ang_vel_reward = np.exp(-0.5 * (3.0**2 / 3.0))
    np.testing.assert_allclose(info["reward_root_orientation"], expected_rot_reward, atol=1e-6)
    np.testing.assert_allclose(info["reward_root_ang_vel"], expected_ang_vel_reward, atol=1e-6)
    np.testing.assert_allclose(total_reward, 0.5 * (expected_rot_reward + expected_ang_vel_reward), atol=1e-6)
    np.testing.assert_allclose(info["err_root_rot"], angle, atol=1e-6)
    np.testing.assert_allclose(info["err_root_ang_vel"], 3.0, atol=1e-6)
    np.testing.assert_allclose(info["root_ang_vel"], 4.0, atol=1e-6)
    np.testing.assert_allclose(info["ref_root_ang_vel"], 1.0, atol=1e-6)


# =====================================================================
# Tests for optional absolute site reward
# =====================================================================


def test_absolute_site_reward_resolves_site_id_and_stores_weight():
    """Absolute site reward configuration should resolve site names to model IDs."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    traj_data = make_traj_data([0.0] * 8)
    th = FakeTrajectoryHandler(traj_data)
    env = make_env(model, th)

    reward = MimicReward(
        env,
        absolute_site_reward_sites=["child_mimic"],
        absolute_site_w_sum=0.1,
        absolute_site_w_exp=10.0,
    )

    np.testing.assert_array_equal(reward._absolute_site_ids, [2])
    assert reward._absolute_site_w_sum == 0.1
    assert reward._absolute_site_w_exp == 10.0
    assert reward._absolute_site_names == ["child_mimic"]


def test_absolute_site_reward_missing_site_raises_value_error():
    """Missing absolute site reward site names should fail during reward initialization."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    traj_data = make_traj_data([0.0] * 8)
    th = FakeTrajectoryHandler(traj_data)
    env = make_env(model, th)

    with pytest.raises(ValueError, match="absolute site reward site not found"):
        MimicReward(
            env,
            absolute_site_reward_sites=["missing_site"],
            absolute_site_w_sum=0.1,
        )


def test_absolute_site_reward_decreases_when_current_site_is_offset():
    """Absolute site reward should be lower when current absolute site position is offset."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)

    init_qpos = np.array([10.0, 5.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    current_qpos = np.array([10.0, 5.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    init_data = make_traj_data(init_qpos, backend=np)
    current_data = make_traj_data(current_qpos, backend=np)
    current_data.site_xpos = np.array(
        [
            [10.0, 5.0, 0.0],
            [10.0, 5.0, 0.1],
            [10.25, 5.0, 0.2],
        ]
    )
    th = FakeTrajectoryHandler(current_data, init_data)
    env = make_env(model, th)

    reward = MimicReward(
        env,
        qpos_w_sum=0.0,
        qvel_w_sum=0.0,
        root_pos_w_sum=0.0,
        root_vel_w_sum=0.0,
        rpos_w_sum=0.0,
        rquat_w_sum=0.0,
        rvel_w_sum=0.0,
        absolute_site_reward_sites=["child_mimic"],
        absolute_site_w_sum=1.0,
        absolute_site_w_exp=10.0,
    )
    carry = make_carry(subtraj_step_no=10, subtraj_step_no_init=0)
    carry = carry.replace(qvel_w_sum=0.0, root_vel_w_sum=0.0)

    sim_qpos = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    matched_data = make_sim_data(sim_qpos, backend=np)
    matched_data.site_xpos = current_data.site_xpos - np.array([10.0, 5.0, 0.0])

    offset_data = make_sim_data(sim_qpos, backend=np)
    offset_data.site_xpos = matched_data.site_xpos.copy()
    offset_data.site_xpos[2] += np.array([0.5, 0.0, 0.0])

    matched_reward, _, matched_info = reward(
        state=np.zeros(10),
        action=np.zeros(3),
        next_state=np.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=matched_data,
        carry=carry,
        backend=np,
    )
    offset_reward, _, offset_info = reward(
        state=np.zeros(10),
        action=np.zeros(3),
        next_state=np.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=offset_data,
        carry=carry,
        backend=np,
    )

    np.testing.assert_allclose(matched_info["reward_absolute_site"], 1.0, atol=1e-6)
    np.testing.assert_allclose(offset_info["err_absolute_site"], np.sqrt((0.5**2 + 0.0 + 0.0) / 3), atol=1e-6)
    assert offset_info["reward_absolute_site"] < matched_info["reward_absolute_site"]
    assert offset_info["err_absolute_site"] > matched_info["err_absolute_site"]
    assert offset_reward < matched_reward


def test_absolute_site_reward_default_off_matches_base_reward():
    """Configuring absolute site names with zero weight should not change reward output."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    traj_qpos = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    traj_data = make_traj_data(traj_qpos, backend=np)
    traj_data.site_xpos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.2],
        ]
    )
    th = FakeTrajectoryHandler(traj_data)
    env = make_env(model, th)

    base_reward = MimicReward(env)
    disabled_absolute_reward = MimicReward(
        env,
        absolute_site_reward_sites=["child_mimic"],
        absolute_site_w_sum=0.0,
    )

    sim_data = make_sim_data(traj_qpos, backend=np)
    sim_data.site_xpos = traj_data.site_xpos.copy()
    carry = make_carry(include_init=False)

    base_total, _, _ = base_reward(
        state=np.zeros(10),
        action=np.zeros(3),
        next_state=np.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=sim_data,
        carry=carry,
        backend=np,
    )
    disabled_total, _, _ = disabled_absolute_reward(
        state=np.zeros(10),
        action=np.zeros(3),
        next_state=np.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=sim_data,
        carry=make_carry(include_init=False),
        backend=np,
    )

    np.testing.assert_allclose(disabled_total, base_total, atol=1e-6)
