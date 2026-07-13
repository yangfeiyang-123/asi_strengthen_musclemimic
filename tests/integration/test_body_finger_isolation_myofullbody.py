from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from musclemimic.core.wrappers.finger_isolation import (
    BodyFingerIsolationWrapper,
    build_named_observation_schema,
    model_action_names,
)
from musclemimic.environments.humanoids.myofullbody import MyoFullBody
from musclemimic.utils.finger_isolation import finger_joint_side


MIMIC_SITES = [
    "pelvis_mimic",
    "upper_body_mimic",
    "head_mimic",
    "left_shoulder_mimic",
    "left_elbow_mimic",
    "left_hand_mimic",
    "right_shoulder_mimic",
    "right_elbow_mimic",
    "right_hand_mimic",
    "left_hip_mimic",
    "left_knee_mimic",
    "left_ankle_mimic",
    "left_toes_mimic",
    "right_hip_mimic",
    "right_knee_mimic",
    "right_ankle_mimic",
    "right_toes_mimic",
]


def _make_env(*, disable_fingers: bool, reward: bool = False):
    kwargs = {
        "disable_fingers": disable_fingers,
        "enable_muscle_length_observations": True,
        "enable_muscle_velocity_observations": True,
        "enable_muscle_force_observations": True,
        "enable_muscle_excitation_observations": True,
        "enable_muscle_activation_observations": True,
        "enable_touch_sensor_observations": True,
        "goal_type": "GoalTrajMimic",
        "goal_params": {
            "n_step_lookahead": 5,
            "n_step_stride": 20,
            "upper_body_xml_name": "torso",
            "enable_motion_phase": True,
            "use_concise_lookahead": True,
            "sites_for_mimic": MIMIC_SITES,
        },
    }
    if reward:
        kwargs.update(
            {
                "reward_type": "MimicReward",
                "reward_params": {
                    "sites_for_mimic": MIMIC_SITES,
                    "exclude_finger_joints": True,
                },
            }
        )
    return MyoFullBody(**kwargs)


def test_real_stage1r_interface_exactly_matches_canonical_stage1():
    full_env = _make_env(disable_fingers=False, reward=True)
    wrapped = BodyFingerIsolationWrapper(
        full_env,
        {
            "expected_partition": [354, 31, 31],
            "expected_removed_observation_dim": 390,
            "expected_policy_observation_dim": 2418,
            "right_grip_provider": {"mode": "constant", "value": 0.0},
            "left_neutral_value": 0.0,
        },
    )
    canonical_env = _make_env(disable_fingers=True)
    canonical_schema = build_named_observation_schema(canonical_env)

    assert wrapped.info.observation_space.shape == canonical_env.info.observation_space.shape == (2418,)
    assert wrapped.info.action_space.shape == canonical_env.info.action_space.shape == (354,)
    assert wrapped.observation_filter.target_schema.schema_hash == canonical_schema.schema_hash
    assert wrapped.policy_actuator_names == model_action_names(canonical_env)
    # CPU MyoFullBody must retain its native reset/step API instead of being
    # accidentally converted to the LocoMjxWrapper API.
    assert wrapped.env is full_env

    full_action = jax.jit(wrapped.expand_body_action)(jnp.zeros(354, dtype=jnp.float32))
    assert full_action.shape == (416,)
    np.testing.assert_allclose(full_action[wrapped.partition.right_grip_indices], 0.0)
    np.testing.assert_allclose(full_action[wrapped.partition.left_neutral_indices], 0.0)

    reward = full_env._reward_function
    included_qpos = set(np.asarray(reward._qpos_ind, dtype=int).tolist())
    included_qvel = set(np.asarray(reward._qvel_ind, dtype=int).tolist())
    for joint_id in range(full_env._model.njnt):
        name = mujoco.mj_id2name(full_env._model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if finger_joint_side(name) is None:
            continue
        assert int(full_env._model.jnt_qposadr[joint_id]) not in included_qpos
        assert int(full_env._model.jnt_dofadr[joint_id]) not in included_qvel

    wrist_id = mujoco.mj_name2id(
        full_env._model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_flexion_r"
    )
    assert int(full_env._model.jnt_qposadr[wrist_id]) in included_qpos
    assert int(full_env._model.jnt_dofadr[wrist_id]) in included_qvel
