"""Structural and physical checks for the composed two-player rally scene."""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from environment.double_play.src.build_double_play_scene import (
    DOUBLE_READY_KEYFRAME,
    default_double_play_scene_path,
    mirror_free_joint_qpos,
)

SCENE_XML = default_double_play_scene_path()

pytestmark = pytest.mark.skipif(
    not SCENE_XML.is_file(),
    reason="double-play scene XML not built; run environment.double_play.src.build_double_play_scene",
)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE_XML))


def _ready_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, DOUBLE_READY_KEYFRAME)
    assert key_id >= 0
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return data


def _geom(model: mujoco.MjModel, name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert geom_id >= 0, f"missing geom {name}"
    return geom_id


def test_two_symmetric_players_one_shuttle(model: mujoco.MjModel) -> None:
    assert model.nu == 708  # 354 muscles per player
    p2_actuators = sum(
        1
        for i in range(model.nu)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith("p2_")
    )
    assert p2_actuators == 354

    for name in (
        "root",
        "p2_root",
        "overall_racket",
        "p2_overall_racket",
        "overall_shuttle",
        "thirdmc_r",
        "p2_thirdmc_r",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0, name
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "p2_overall_shuttle") < 0

    for site in (
        "overall_stringbed_center_site",
        "p2_overall_stringbed_center_site",
        "overall_cork_contact_site",
        "rh_palm_grip_site",
        "p2_rh_palm_grip_site",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) >= 0, site

    # neither racket has a joint: both are jointless exact children
    for racket in ("overall_racket", "p2_overall_racket"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, racket)
        assert int(model.body_jntnum[body]) == 0


def test_contact_bits_isolate_hands_and_stringbeds(model: mujoco.MjModel) -> None:
    cork = _geom(model, "overall_cork_collision")
    assert int(model.geom_conaffinity[cork]) == 13  # 1|4|8: floor/net + rackets + compat

    for proxy_name in (
        "overall_stringbed_ground_contact_proxy",
        "p2_overall_stringbed_ground_contact_proxy",
    ):
        proxy = _geom(model, proxy_name)
        assert int(model.geom_contype[proxy]) == 16
        assert int(model.geom_conaffinity[proxy]) == 16

    floor = _geom(model, "floor")
    assert int(model.geom_conaffinity[floor]) & 16  # racket faces can rest on the floor

    # both racket frames share bit 4 and never meet human bit 1
    for frame_geom in ("overall_head_frame_00", "p2_overall_head_frame_00"):
        geom_id = _geom(model, frame_geom)
        assert int(model.geom_contype[geom_id]) == 4
        assert int(model.geom_conaffinity[geom_id]) == 4

    # p2's compat ellipsoids got the same MJX treatment as p1's
    ellipsoid = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
    stray = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        for gid in range(model.ngeom)
        if int(model.geom_type[gid]) == ellipsoid
        and (int(model.geom_contype[gid]), int(model.geom_conaffinity[gid])) not in {(0, 0), (8, 0)}
    ]
    assert stray == []


def test_double_ready_keyframe_is_mirror_symmetric(model: mujoco.MjModel) -> None:
    data = _ready_data(model)
    roots = {}
    for name in ("root", "p2_root"):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = int(model.jnt_qposadr[joint])
        roots[name] = np.asarray(data.qpos[adr : adr + 7], dtype=float)
    np.testing.assert_allclose(roots["p2_root"], mirror_free_joint_qpos(roots["root"]), atol=1e-12)
    assert roots["root"][0] < -3.9  # both in their backcourts
    assert roots["p2_root"][0] > 3.9

    # rackets sit identically relative to each player's palm
    distances = []
    for racket, palm in (
        ("overall_racket", "rh_palm_grip_site"),
        ("p2_overall_racket", "p2_rh_palm_grip_site"),
    ):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, racket)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, palm)
        distances.append(float(np.linalg.norm(data.xpos[body] - data.site_xpos[site])))
    assert distances[0] == pytest.approx(distances[1], abs=1e-9)
    assert distances[0] < 0.12

    # mirrored stringbed centers
    p1_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "overall_stringbed_center_site")
    p2_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "p2_overall_stringbed_center_site")
    p1_pos = np.asarray(data.site_xpos[p1_site], dtype=float)
    p2_pos = np.asarray(data.site_xpos[p2_site], dtype=float)
    np.testing.assert_allclose(p2_pos, [-p1_pos[0], -p1_pos[1], p1_pos[2]], atol=1e-9)


def test_passive_scene_is_stable(model: mujoco.MjModel) -> None:
    data = _ready_data(model)
    data.ctrl[:] = 0.0
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert bool(np.isfinite(data.qacc).all())
    for name in ("root", "p2_root"):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = int(model.jnt_qposadr[joint])
        assert float(data.qpos[adr + 2]) > 0.8  # nobody exploded or sank
