"""Tests for the v2 racket-shuttle contact extensions.

Covers: the swept plane-crossing detector (unit + end-to-end tunneling
recovery), the eccentric cork angular-impulse closure, and speed-dependent
event restitution.  Legacy defaults must leave all of it off.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.badminton_physics import (  # noqa: E402
    BadmintonPhysics,
    BadmintonPhysicsConfig,
)
from environment.overall_environment.src.paths import default_incoming_scene_path  # noqa: E402
from environment.racket.src.racket_stringbed import (  # noqa: E402
    RacketGeometry,
    swept_stringbed_crossing,
)
from environment.shuttlecock.src.shuttlecock_aero import (  # noqa: E402
    SHUTTLE_DIAG_INERTIA_KG_M2,
    restore_shuttle_inertia,
)
from environment.shuttlecock.src.shuttlecock_racket_impact import (  # noqa: E402
    ShuttlecockImpactConfig,
    compute_cork_angular_impulse_omega,
    effective_normal_restitution,
)

SCENE_XML = default_incoming_scene_path()

pytestmark = pytest.mark.skipif(
    not SCENE_XML.is_file(),
    reason="incoming scene XML not built; run environment.overall_environment.src.incoming_scene",
)

CORK_LOCAL_OFFSET = np.array([0.0, 0.0, 0.011154])


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE_XML))


def _ready_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return data


def _shuttle_addresses(model: mujoco.MjModel) -> tuple[int, int]:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "overall_shuttle_free")
    return int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])


def _racket_frame(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    racket = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "overall_racket")
    origin = np.asarray(data.xpos[racket], dtype=float)
    rot = np.asarray(data.xmat[racket], dtype=float).reshape(3, 3)
    return origin, rot


def _set_shuttle(model, data, pos, quat, vel) -> None:
    qadr, dadr = _shuttle_addresses(model)
    data.qpos[qadr : qadr + 3] = pos
    data.qpos[qadr + 3 : qadr + 7] = quat
    data.qvel[dadr : dadr + 3] = vel
    data.qvel[dadr + 3 : dadr + 6] = 0.0
    mujoco.mj_forward(model, data)


# ---- swept_stringbed_crossing unit tests -----------------------------------


def test_swept_crossing_detects_center_plane_pierce():
    geom = RacketGeometry()
    prev = np.array([0.0, geom.stringbed_center_y, 0.03])
    curr = np.array([0.0, geom.stringbed_center_y, -0.04])
    result = swept_stringbed_crossing(prev, curr, geom)
    assert result["crossed"] is True
    assert result["side_from"] == 1.0
    assert result["fraction"] == pytest.approx(3.0 / 7.0)
    assert result["rho2"] == pytest.approx(0.0, abs=1e-12)


def test_swept_crossing_ignores_same_side_and_outside_paths():
    geom = RacketGeometry()
    center_y = geom.stringbed_center_y
    same_side = swept_stringbed_crossing(
        np.array([0.0, center_y, 0.05]), np.array([0.0, center_y, 0.01]), geom
    )
    assert same_side["crossed"] is False

    outside = swept_stringbed_crossing(
        np.array([0.5, center_y, 0.03]), np.array([0.5, center_y, -0.03]), geom
    )
    assert outside["crossed"] is False
    assert outside["reason"] == "outside_stringbed"

    from_below = swept_stringbed_crossing(
        np.array([0.0, center_y, -0.02]), np.array([0.0, center_y, 0.02]), geom
    )
    assert from_below["crossed"] is True
    assert from_below["side_from"] == -1.0


# ---- end-to-end tunneling recovery -----------------------------------------


def _launch_tunneling_cork(model, data, *, closing_speed: float) -> np.ndarray:
    """Aim the cork at the stringbed center, fast enough to jump the band."""
    origin, rot = _racket_frame(model, data)
    normal = rot[:, 2]
    stringbed_center = origin + rot @ np.array([0.0, 0.532, 0.0])
    cork_start = stringbed_center + 0.025 * normal  # outside the 15 mm band
    shuttle_pos = cork_start - CORK_LOCAL_OFFSET  # identity quat cork offset
    _set_shuttle(model, data, shuttle_pos, np.array([1.0, 0.0, 0.0, 0.0]), -closing_speed * normal)
    return normal


def test_high_speed_impact_tunnels_with_legacy_config(model):
    data = _ready_data(model)
    normal = _launch_tunneling_cork(model, data, closing_speed=60.0)
    physics = BadmintonPhysics()  # legacy: no swept detection
    used_event = False
    for _ in range(4):
        diag = physics.substep(model, data)
        used_event = used_event or bool(diag["event_rebound_used"])
    assert used_event is False
    _, dadr = _shuttle_addresses(model)
    velocity = np.asarray(data.qvel[dadr : dadr + 3], dtype=float)
    assert float(np.dot(velocity, normal)) < -30.0  # still flying through the bed


def test_swept_detection_recovers_tunneled_impact(model):
    data = _ready_data(model)
    normal = _launch_tunneling_cork(model, data, closing_speed=60.0)
    physics = BadmintonPhysics(BadmintonPhysicsConfig(enable_swept_crossing_detection=True))
    diags = [physics.substep(model, data) for _ in range(4)]
    swept_hits = [d for d in diags if d["swept_crossing_used"]]
    assert len(swept_hits) == 1
    assert swept_hits[0]["event_rebound_used"] is True
    _, dadr = _shuttle_addresses(model)
    velocity = np.asarray(data.qvel[dadr : dadr + 3], dtype=float)
    assert float(np.dot(velocity, normal)) > 1.0  # reflected back out
    # equal/opposite reaction reached the racket chain
    impulse_shuttle = swept_hits[0]["event_impulse_on_shuttle_world_ns"]
    impulse_racket = swept_hits[0]["event_impulse_on_racket_world_ns"]
    np.testing.assert_allclose(impulse_shuttle + impulse_racket, 0.0, atol=1e-12)


def test_swept_detection_leaves_slow_contact_to_the_spring(model):
    """A slow bed touch must keep using the continuous penalty spring."""
    data = _ready_data(model)
    origin, rot = _racket_frame(model, data)
    normal = rot[:, 2]
    stringbed_center = origin + rot @ np.array([0.0, 0.532, 0.0])
    cork_start = stringbed_center + 0.005 * normal  # inside the band
    _set_shuttle(
        model,
        data,
        cork_start - CORK_LOCAL_OFFSET,
        np.array([1.0, 0.0, 0.0, 0.0]),
        -1.0 * normal,
    )
    physics = BadmintonPhysics(BadmintonPhysicsConfig(enable_swept_crossing_detection=True))
    diag = physics.substep(model, data)
    assert diag["event_rebound_used"] is False
    assert diag["swept_crossing_used"] is False
    assert bool(diag["stringbed"]["active"]) is True


# ---- cork angular-impulse closure ------------------------------------------


def test_cork_angular_impulse_math_matches_rigid_body_formula():
    cfg = ShuttlecockImpactConfig()
    omega_before = np.array([1.0, -2.0, 0.5])
    impulse = np.array([0.0, 0.2, 0.0])
    contact = np.array([0.0, 0.0, 1.011154])
    com = np.array([0.0, 0.0, 1.0])
    inertia = np.array([3.32e-6, 3.32e-6, 1.06e-6])
    rot = np.eye(3)
    omega_after = compute_cork_angular_impulse_omega(
        omega_before_world=omega_before,
        impulse_on_shuttle_world=impulse,
        contact_point_world=contact,
        com_world=com,
        inertia_diag_body_kg_m2=inertia,
        inertia_rot_world=rot,
        cfg=cfg,
    )
    expected_unclipped = omega_before + np.cross(contact - com, impulse) / inertia
    expected = expected_unclipped * min(
        1.0, cfg.max_shuttle_angular_velocity_rad_s / np.linalg.norm(expected_unclipped)
    )
    np.testing.assert_allclose(omega_after, expected, rtol=1e-12)
    assert float(np.linalg.norm(omega_after)) <= cfg.max_shuttle_angular_velocity_rad_s + 1e-9


def _launch_eccentric_impact(model, data, *, closing_speed: float) -> None:
    """Cork on the bed with the nose axis parallel to the face (max lever arm)."""
    origin, rot = _racket_frame(model, data)
    normal = rot[:, 2]
    lateral = rot[:, 0]
    stringbed_center = origin + rot @ np.array([0.0, 0.532, 0.0])
    cork_target = stringbed_center + 0.005 * normal
    # orient shuttle +z (nose) along the racket lateral axis
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, lateral)
    axis /= max(np.linalg.norm(axis), 1e-12)
    angle = float(np.arccos(np.clip(np.dot(z_axis, lateral), -1.0, 1.0)))
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, axis, angle)
    rot_shuttle = np.zeros(9)
    mujoco.mju_quat2Mat(rot_shuttle, quat)
    cork_world_offset = rot_shuttle.reshape(3, 3) @ CORK_LOCAL_OFFSET
    _set_shuttle(model, data, cork_target - cork_world_offset, quat, -closing_speed * normal)


def test_composed_scene_clamps_shuttle_inertia_and_restore_fixes_it(model):
    """The MyoFullBody base spec's boundinertia=1e-4 inflates the 3.32e-6 shuttle inertia 30x."""
    shuttle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "overall_shuttle")
    np.testing.assert_allclose(model.body_inertia[shuttle], 1.0e-4)

    fresh = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    restore_shuttle_inertia(fresh, "overall_shuttle")
    np.testing.assert_allclose(fresh.body_inertia[shuttle], SHUTTLE_DIAG_INERTIA_KG_M2)


def test_cork_angular_impulse_spins_the_shuttle_on_event():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    restore_shuttle_inertia(model, "overall_shuttle")
    _, dadr = _shuttle_addresses(model)

    data = _ready_data(model)
    _launch_eccentric_impact(model, data, closing_speed=8.0)
    legacy = BadmintonPhysics()
    diag_legacy = legacy.substep(model, data)
    assert diag_legacy["event_rebound_used"] is True
    omega_legacy = float(np.linalg.norm(data.qvel[dadr + 3 : dadr + 6]))

    data = _ready_data(model)
    _launch_eccentric_impact(model, data, closing_speed=8.0)
    v2 = BadmintonPhysics(BadmintonPhysicsConfig(apply_cork_angular_impulse=True))
    diag_v2 = v2.substep(model, data)
    assert diag_v2["event_rebound_used"] is True
    omega_v2 = float(np.linalg.norm(data.qvel[dadr + 3 : dadr + 6]))
    omega_diag = float(np.linalg.norm(diag_v2["event_shuttle_omega_after_world_rad_s"]))

    cfg = ShuttlecockImpactConfig()
    assert omega_legacy < 1.0  # legacy leaves the angular channel untouched
    assert omega_v2 > 100.0  # the eccentric impulse flips the shuttle hard
    assert omega_v2 <= cfg.max_shuttle_angular_velocity_rad_s + 1e-6
    # The same substep still integrates one tick of aero torque after the set.
    # With the physical inertia actually in effect (restore_shuttle_inertia now
    # refreshes the model constants), that tick moves omega by about
    # damping_torque/I*dt ~ 1.3 rad/s at 200 rad/s, hence the 1e-2 budget.
    assert omega_diag == pytest.approx(omega_v2, rel=1e-2)


# ---- speed-dependent restitution -------------------------------------------


def test_effective_restitution_legacy_slope_zero_is_constant():
    cfg = ShuttlecockImpactConfig()
    for speed in (1.0, 10.0, 50.0, 200.0):
        assert effective_normal_restitution(closing_speed_m_s=speed, cfg=cfg) == pytest.approx(
            cfg.event_restitution_normal
        )


def test_effective_restitution_drops_with_speed_and_floors():
    cfg = ShuttlecockImpactConfig(
        restitution_speed_slope_per_m_s=0.005,
        restitution_reference_speed_m_s=10.0,
        min_restitution=0.30,
    )
    assert effective_normal_restitution(closing_speed_m_s=5.0, cfg=cfg) == pytest.approx(0.5)
    assert effective_normal_restitution(closing_speed_m_s=30.0, cfg=cfg) == pytest.approx(0.4)
    assert effective_normal_restitution(closing_speed_m_s=500.0, cfg=cfg) == pytest.approx(0.30)


def test_legacy_config_defaults_keep_v2_features_off():
    cfg = BadmintonPhysicsConfig()
    assert cfg.enable_swept_crossing_detection is False
    assert cfg.apply_cork_angular_impulse is False
    impact = ShuttlecockImpactConfig()
    assert impact.restitution_speed_slope_per_m_s == 0.0
