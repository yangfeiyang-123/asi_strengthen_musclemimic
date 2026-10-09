from __future__ import annotations

import mujoco
import numpy as np
import pytest

from loco_mujoco.core.observations import ObservationType
from musclemimic.environments.humanoids.myofullbody import (
    MyoFullBody,
    remove_finger_dofs,
)


@pytest.fixture(scope="module")
def real_myofullbody_env() -> MyoFullBody:
    return MyoFullBody(
        disable_fingers=True,
        enable_joint_pos_observations=False,
        enable_joint_vel_observations=False,
        enable_touch_sensor_observations=False,
    )


def test_real_myofullbody_spec_and_runtime_use_unit_muscle_control_range(
    real_myofullbody_env: MyoFullBody,
) -> None:
    spec_muscles = [
        actuator
        for actuator in real_myofullbody_env._mjspec.actuators
        if actuator.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    ]
    assert spec_muscles
    for actuator in spec_muscles:
        np.testing.assert_allclose(actuator.ctrlrange, [0.0, 1.0])
        assert actuator.ctrllimited

    model = real_myofullbody_env._model
    muscle_ids = np.flatnonzero(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
    assert muscle_ids.size
    np.testing.assert_allclose(
        model.actuator_ctrlrange[muscle_ids],
        np.tile([0.0, 1.0], (muscle_ids.size, 1)),
    )
    assert np.all(model.actuator_ctrllimited[muscle_ids])
    assert np.all(model.actuator_actadr[muscle_ids] >= 0)
    assert np.all(model.actuator_actnum[muscle_ids] == 1)

    np.testing.assert_allclose(
        real_myofullbody_env.info.action_space.low,
        -np.ones(model.nu),
    )
    np.testing.assert_allclose(
        real_myofullbody_env.info.action_space.high,
        np.ones(model.nu),
    )


def test_low_policy_action_maps_to_nonzero_excitation_without_dead_zone(
    real_myofullbody_env: MyoFullBody,
) -> None:
    model = real_myofullbody_env._model
    policy_action = np.full(model.nu, -0.5)
    excitation = real_myofullbody_env._control_func._unnormalize_action(policy_action)
    np.testing.assert_allclose(excitation, 0.25)

    data = mujoco.MjData(model)
    data.qpos[:] = real_myofullbody_env._data.qpos
    data.ctrl[:] = excitation
    mujoco.mj_step(model, data)
    assert np.all(data.act > 0.0)


def test_finger_removal_uses_exact_actuator_targets_for_short_op_name() -> None:
    spec = mujoco.MjSpec.from_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <joint name="joint" type="hinge"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 0 0 1"/>
            </body>
          </worldbody>
          <tendon>
            <fixed name="thumb_opposition_path">
              <joint joint="joint" coef="1"/>
            </fixed>
            <fixed name="POP_decoy">
              <joint joint="joint" coef="1"/>
            </fixed>
          </tendon>
          <actuator>
            <general name="OP" tendon="thumb_opposition_path"/>
            <general name="KEEP" tendon="POP_decoy"/>
          </actuator>
        </mujoco>
        """
    )

    remove_finger_dofs(spec)

    assert [actuator.name for actuator in spec.actuators] == ["KEEP"]
    assert [tendon.name for tendon in spec.tendons] == ["POP_decoy"]


def test_optional_muscle_observations_filter_nonmuscle_and_invalid_activation() -> None:
    spec = mujoco.MjSpec.from_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <joint name="joint" type="hinge"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 0 0 1"/>
            </body>
          </worldbody>
          <tendon>
            <fixed name="muscle_path">
              <joint joint="joint" coef="1"/>
            </fixed>
            <fixed name="invalid_muscle_path">
              <joint joint="joint" coef="1"/>
            </fixed>
          </tendon>
          <actuator>
            <muscle name="muscle" tendon="muscle_path" lengthrange="0.1 1.0"/>
            <muscle name="invalid_muscle" tendon="invalid_muscle_path" lengthrange="0.1 1.0"/>
            <motor name="motor" joint="joint"/>
          </actuator>
        </mujoco>
        """
    )
    spec.actuator("invalid_muscle").actdim = 0

    env = MyoFullBody.__new__(MyoFullBody)
    env._enable_joint_pos_observations = False
    env._enable_joint_vel_observations = False
    env._enable_muscle_length_observations = True
    env._enable_muscle_velocity_observations = True
    env._enable_muscle_force_observations = True
    env._enable_muscle_excitation_observations = True
    env._enable_muscle_activation_observations = True
    env._enable_touch_sensor_observations = False

    observations = env._get_observation_specification(spec)
    names_by_type: dict[str, set[str]] = {}
    for observation in observations:
        names_by_type.setdefault(type(observation).__name__, set()).add(observation.xml_name)

    expected_muscles = {"muscle", "invalid_muscle"}
    assert names_by_type["ActuatorLength"] == expected_muscles
    assert names_by_type["ActuatorVelocity"] == expected_muscles
    assert names_by_type["ActuatorForce"] == expected_muscles
    assert names_by_type["ActuatorExcitation"] == expected_muscles
    assert names_by_type["ActuatorActivation"] == {"muscle"}


def test_muscle_observations_use_ctrlrange_and_packed_activation_address() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <joint name="joint" type="hinge"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 0 0 1"/>
            </body>
          </worldbody>
          <tendon>
            <fixed name="muscle_path">
              <joint joint="joint" coef="1"/>
            </fixed>
          </tendon>
          <actuator>
            <motor name="motor" joint="joint" ctrllimited="true"
                   ctrlrange="-2 2"/>
            <muscle name="muscle" tendon="muscle_path"
                    ctrlrange="0 1" lengthrange="0.1 1.0"/>
          </actuator>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    muscle_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "muscle",
    )
    assert muscle_id == 1
    assert model.actuator_actadr[muscle_id] == 0

    excitation = ObservationType.ActuatorExcitation(
        "excitation",
        xml_name="muscle",
    )
    excitation._init_from_mj(None, model, data, 0)
    np.testing.assert_array_equal(excitation.data_type_ind, [muscle_id])
    np.testing.assert_allclose(excitation.min, [0.0])
    np.testing.assert_allclose(excitation.max, [1.0])

    activation = ObservationType.ActuatorActivation(
        "activation",
        xml_name="muscle",
    )
    activation._init_from_mj(None, model, data, 0)
    np.testing.assert_array_equal(
        activation.data_type_ind,
        [model.actuator_actadr[muscle_id]],
    )
