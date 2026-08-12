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


def test_racket_butt_at_grip_frame():
    """The racket butt cap must sit at the grip pose on the palm, not float at
    the asset's world spawn offset (regression: the asset's ``pos="0 0 1.2"``
    root offset leaked into the rigid attachment)."""
    from musclemimic.environments.humanoids.myofullbody_racket import (
        DEFAULT_RACKET_GRIP_POS,
        DEFAULT_RACKET_GRIP_QUAT,
    )

    env = MyoFullBodyRacket(disable_fingers=True)
    assert DEFAULT_RACKET_GRIP_POS == pytest.approx(
        (-0.01746, 0.01085, -0.05332),
        abs=1e-9,
    )
    assert DEFAULT_RACKET_GRIP_QUAT == pytest.approx(
        (0.42064706, 0.486617281, 0.05363487, -0.763795112),
        abs=1e-9,
    )
    m = env._model
    rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, RACKET_BODY_NAME)
    # local offset == grip pos exactly (racket root pose zeroed before attach)
    assert np.allclose(m.body_pos[rid], DEFAULT_RACKET_GRIP_POS, atol=1e-6)

    d = mujoco.MjData(m)
    d.qpos[:] = env._data.qpos
    mujoco.mj_forward(m, d)
    hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "thirdmc_r")
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "racket_butt_site")
    assert sid >= 0
    # butt cap within a hand's breadth of the palm body
    assert np.linalg.norm(d.site_xpos[sid] - d.xpos[hid]) < 0.12


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


def test_racket_mimic_reward_registered():
    from loco_mujoco.core.reward.base import Reward

    from musclemimic.core.reward.trajectory_based import RacketMimicReward  # noqa: F401

    assert "RacketMimicReward" in Reward.registered


def _make_racket_reward_env():
    env = MyoFullBodyRacket(
        disable_fingers=True,
        reward_type="RacketMimicReward",
        reward_params=dict(
            racket_pos_w_sum=0.3,
            racket_pos_w_exp=50.0,
            racket_rot_w_sum=0.15,
            racket_rot_w_exp=5.0,
        ),
    )
    return env, env._reward_function


def test_racket_reward_reference_matches_fk():
    """The derived reference racket pose (hand site pose ∘ fixed grip transform)
    must equal forward kinematics of the racket site for arbitrary poses."""
    env, reward = _make_racket_reward_env()
    m = env._model
    d = mujoco.MjData(m)
    hand_sid = reward._racket_hand_site_id
    racket_sid = reward._racket_site_id

    rng = np.random.default_rng(42)
    for _ in range(10):
        d.qpos[:] = env._data.qpos
        d.qpos[7:] += rng.uniform(-0.4, 0.4, size=m.nq - 7)
        mujoco.mj_forward(m, d)
        pred_pos, pred_mat = reward.derive_reference_racket_pose(
            d.site_xpos[hand_sid], d.site_xmat[hand_sid].reshape(3, 3)
        )
        assert np.allclose(pred_pos, d.site_xpos[racket_sid], atol=1e-10)
        assert np.allclose(pred_mat, d.site_xmat[racket_sid].reshape(3, 3), atol=1e-10)


def test_racket_reward_rejects_bare_hand_env():
    """Without the racket the reward must fail loudly, not silently track nothing."""
    with pytest.raises(ValueError, match="racket site"):
        MyoFullBody(
            disable_fingers=True,
            reward_type="RacketMimicReward",
            reward_params=dict(racket_pos_w_sum=0.3),
        )


def test_fingers_enabled_dims_and_action_space():
    """Fingers on must yield the full-body dims that match the Stage-3 hit env."""
    m = MyoFullBodyRacket(disable_fingers=False)._model
    assert (m.nq, m.nv, m.nu) == (129, 128, 416)


def test_racket_mass_scale():
    from musclemimic.environments.humanoids.myofullbody_racket import RACKET_BODY_NAME

    base = MyoFullBodyRacket(disable_fingers=True)._model
    scaled = MyoFullBodyRacket(disable_fingers=True, racket_mass_scale=0.5)._model
    rid = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, RACKET_BODY_NAME)
    assert float(scaled.body_mass[rid]) == pytest.approx(0.5 * float(base.body_mass[rid]), rel=1e-6)
    with pytest.raises(ValueError, match="mass_scale"):
        MyoFullBodyRacket(disable_fingers=True, racket_mass_scale=0.0)


def test_grip_finger_reference_present_with_fingers():
    """The grip reference must resolve to the model's right-hand finger joints."""
    env = MyoFullBodyRacket(disable_fingers=False)
    from musclemimic.badminton.racket_grip_preset import (
        load_racket_grip_preset,
    )

    default_preset = load_racket_grip_preset(
        "configs/racket_grip/forehand_clear_grip_v2_custom.json"
    )
    assert env.racket_grip_preset_fingerprint == default_preset.fingerprint
    names = env.grip_finger_names
    addrs = env.grip_finger_qpos_addrs
    targets = env.grip_finger_targets
    assert len(names) == 20
    assert addrs.shape == targets.shape == (20,)
    # grip pose is a closed hand, not the zero (open) pose the trajectory implies
    assert float(np.max(np.abs(targets))) > 0.5

    # fingers disabled -> nothing to override
    env_off = MyoFullBodyRacket(disable_fingers=True)
    assert len(env_off.grip_finger_qpos_addrs) == 0


def test_global_grip_preset_applies_to_environment(tmp_path):
    """One preset controls both the racket attachment and reset/reward targets."""
    from environment.overall_environment.src.racket_attachment import (
        load_racket_attachment_contract,
    )
    from musclemimic.badminton.racket_grip_preset import write_racket_grip_preset

    default_env = MyoFullBodyRacket(disable_fingers=False)
    angles = dict(
        zip(
            default_env.grip_finger_names,
            default_env.grip_finger_targets,
            strict=True,
        )
    )
    angles[default_env.grip_finger_names[0]] = float(
        default_env.grip_finger_targets[0] + 0.05
    )
    preset = write_racket_grip_preset(
        tmp_path / "all_trajectories_grip.json",
        preset_id="all_trajectories_grip",
        attachment_contract=load_racket_attachment_contract(),
        finger_joint_angles_rad=angles,
    )

    env = MyoFullBodyRacket(
        disable_fingers=False,
        racket_grip_preset=preset.source_path,
    )
    assert env.racket_grip_preset_fingerprint == preset.fingerprint
    assert env.racket_attachment_contract_fingerprint == (
        preset.attachment_contract_fingerprint
    )
    assert env.grip_finger_targets[0] == pytest.approx(angles[env.grip_finger_names[0]])


def test_grip_init_handler_closes_hand(monkeypatch):
    """RacketGripInitialStateHandler must set finger qpos to the grip pose at reset."""
    import os
    from pathlib import Path

    from loco_mujoco.task_factories import AMASSDatasetConf, ImitationFactory

    # Resolve the retargeted trajectory locally (no network / SMPL fitting).
    repo = Path(__file__).resolve().parents[1]
    clip = repo / "datasets/forehandClear_standard/muscle_trajectory/optimized/6月2日(1)-1.npz"
    if not clip.is_file():
        pytest.skip("local retargeted trajectory not available")
    monkeypatch.setenv("MUSCLEMIMIC_GMR_CACHE_PATH", str(repo / "datasets"))
    monkeypatch.setenv("MUSCLEMIMIC_DATASETS_ROOT", str(repo / "datasets"))
    monkeypatch.setenv("CONVERTED_AMASS_PATH", str(repo / "datasets/_global/muscle_trajectory/gmr_cache"))

    env = ImitationFactory.make(
        "MyoFullBodyRacket",
        amass_dataset_conf=AMASSDatasetConf(
            rel_dataset_path=["forehandClear_standard/muscle_trajectory/optimized/6月2日(1)-1"],
            retargeting_method="gmr",
        ),
        disable_fingers=False,
        init_state_type="RacketGripInitialStateHandler",
        reward_type="RacketMimicReward",
        reward_params=dict(finger_grip_w_sum=0.2, racket_pos_w_sum=0.3, racket_rot_w_sum=0.15),
    )
    env.reset()
    addrs = env.grip_finger_qpos_addrs
    targets = env.grip_finger_targets
    assert np.allclose(env._data.qpos[addrs], targets, atol=1e-6)
    # a step stays finite with the fingers-on action space
    action = np.zeros(env._model.nu)
    env.step(action)
    assert np.all(np.isfinite(env._data.qpos))
