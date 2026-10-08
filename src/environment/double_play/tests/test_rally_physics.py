"""Tests for the two-racket rally substep physics."""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from environment.double_play.src.build_double_play_scene import (
    DOUBLE_READY_KEYFRAME,
    default_double_play_scene_path,
)
from environment.double_play.src.rally_physics import (
    RallyBadmintonPhysics,
    RallyPhysicsConfig,
)
from environment.shuttlecock.src.shuttlecock_aero import restore_shuttle_inertia

SCENE_XML = default_double_play_scene_path()

pytestmark = pytest.mark.skipif(
    not SCENE_XML.is_file(),
    reason="double-play scene XML not built; run environment.double_play.src.build_double_play_scene",
)

CORK_LOCAL_OFFSET = np.array([0.0, 0.0, 0.011154])


@pytest.fixture()
def model() -> mujoco.MjModel:
    loaded = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    restore_shuttle_inertia(loaded, "overall_shuttle")
    return loaded


def _ready_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, DOUBLE_READY_KEYFRAME)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return data


def _shuttle_addresses(model: mujoco.MjModel) -> tuple[int, int]:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "overall_shuttle_free")
    return int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])


def _place_cork_on_stringbed(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    racket_body_name: str,
    *,
    closing_speed: float,
    offset_along_normal_m: float = 0.005,
) -> np.ndarray:
    racket = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, racket_body_name)
    origin = np.asarray(data.xpos[racket], dtype=float)
    rot = np.asarray(data.xmat[racket], dtype=float).reshape(3, 3)
    normal = rot[:, 2]
    stringbed_center = origin + rot @ np.array([0.0, 0.532, 0.0])
    cork_target = stringbed_center + offset_along_normal_m * normal
    qadr, dadr = _shuttle_addresses(model)
    data.qpos[qadr : qadr + 3] = cork_target - CORK_LOCAL_OFFSET
    data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dadr : dadr + 3] = -closing_speed * normal
    data.qvel[dadr + 3 : dadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return normal


@pytest.mark.parametrize("racket", ["overall_racket", "p2_overall_racket"])
def test_event_rebound_fires_for_each_racket(model: mujoco.MjModel, racket: str) -> None:
    data = _ready_data(model)
    normal = _place_cork_on_stringbed(model, data, racket, closing_speed=8.0)
    physics = RallyBadmintonPhysics()
    diag = physics.substep(model, data)
    assert diag["event_racket"] == racket
    assert diag["rackets"][racket]["event_rebound_used"] is True
    other = "p2_overall_racket" if racket == "overall_racket" else "overall_racket"
    assert diag["rackets"][other]["event_rebound_used"] is False
    _, dadr = _shuttle_addresses(model)
    velocity = np.asarray(data.qvel[dadr : dadr + 3], dtype=float)
    assert float(np.dot(velocity, normal)) > 1.0
    # v2 default: the eccentric cork impulse also spun the shuttle
    omega = np.asarray(data.qvel[dadr + 3 : dadr + 6], dtype=float)
    assert float(np.linalg.norm(omega)) > 1.0
    # equal/opposite impulses
    event = diag["event"]
    np.testing.assert_allclose(
        event["impulse_on_shuttle_world_ns"] + event["impulse_on_racket_world_ns"],
        0.0,
        atol=1e-12,
    )


def test_per_racket_cooldown_is_independent(model: mujoco.MjModel) -> None:
    data = _ready_data(model)
    _place_cork_on_stringbed(model, data, "overall_racket", closing_speed=8.0)
    physics = RallyBadmintonPhysics(RallyPhysicsConfig(rebound_cooldown_substeps=50))
    first = physics.substep(model, data)
    assert first["event_racket"] == "overall_racket"

    # re-arm on the same racket: swallowed by that racket's cooldown
    _place_cork_on_stringbed(model, data, "overall_racket", closing_speed=8.0)
    second = physics.substep(model, data)
    assert second["event_racket"] is None
    assert second["rackets"]["overall_racket"]["stringbed_force_suppressed"] is True

    # but the other racket is immediately hot
    _place_cork_on_stringbed(model, data, "p2_overall_racket", closing_speed=8.0)
    third = physics.substep(model, data)
    assert third["event_racket"] == "p2_overall_racket"


def test_swept_crossing_catches_tunneling_on_p2_racket(model: mujoco.MjModel) -> None:
    data = _ready_data(model)
    normal = _place_cork_on_stringbed(
        model, data, "p2_overall_racket", closing_speed=60.0, offset_along_normal_m=0.025
    )
    physics = RallyBadmintonPhysics()
    diags = [physics.substep(model, data) for _ in range(4)]
    swept = [d for d in diags if d["event_racket"] == "p2_overall_racket"]
    assert len(swept) == 1
    assert swept[0]["event"]["swept_crossing_used"] is True
    _, dadr = _shuttle_addresses(model)
    velocity = np.asarray(data.qvel[dadr : dadr + 3], dtype=float)
    assert float(np.dot(velocity, normal)) > 1.0


def test_aero_decelerates_free_flight(model: mujoco.MjModel) -> None:
    qadr, dadr = _shuttle_addresses(model)
    launch_pos = np.array([0.0, 0.0, 4.0])
    launch_vel = np.array([-12.0, 0.0, 3.0])

    data = _ready_data(model)
    data.qpos[qadr : qadr + 3] = launch_pos
    data.qvel[dadr : dadr + 3] = launch_vel
    mujoco.mj_forward(model, data)
    physics = RallyBadmintonPhysics()
    for _ in range(300):
        physics.substep(model, data)
    speed_aero = float(np.linalg.norm(data.qvel[dadr : dadr + 3]))

    data = _ready_data(model)
    data.qpos[qadr : qadr + 3] = launch_pos
    data.qvel[dadr : dadr + 3] = launch_vel
    mujoco.mj_forward(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)
    speed_plain = float(np.linalg.norm(data.qvel[dadr : dadr + 3]))
    assert speed_aero < speed_plain - 2.0
