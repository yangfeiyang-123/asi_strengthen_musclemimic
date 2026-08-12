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
    default_aero_config,
)
from environment.overall_environment.src.paths import default_incoming_scene_path  # noqa: E402
from environment.overall_environment.src.shuttle_feeder import (  # noqa: E402
    integrate_shuttle_flight,
    launch_quat_from_velocity,
    sample_feed,
)
from environment.shuttlecock.src.shuttlecock_aero import (  # noqa: E402
    apply_shuttlecock_aero,
)

SCENE_XML = default_incoming_scene_path()

pytestmark = pytest.mark.skipif(
    not SCENE_XML.is_file(),
    reason="incoming scene XML not built; run environment.overall_environment.src.incoming_scene",
)


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


def _set_shuttle_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pos: np.ndarray,
    quat: np.ndarray,
    vel: np.ndarray,
) -> None:
    qadr, dadr = _shuttle_addresses(model)
    data.qpos[qadr : qadr + 3] = pos
    data.qpos[qadr + 3 : qadr + 7] = quat
    data.qvel[dadr : dadr + 3] = vel
    data.qvel[dadr + 3 : dadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def test_aero_decelerates_shuttle(model: mujoco.MjModel) -> None:
    vel = np.array([-15.0, 0.0, 0.0])
    pos = np.array([4.0, 0.0, 6.0])
    quat = launch_quat_from_velocity(vel)

    data_aero = _ready_data(model)
    _set_shuttle_state(model, data_aero, pos, quat, vel)
    physics = BadmintonPhysics()
    for _ in range(300):
        physics.substep(model, data_aero)

    data_plain = _ready_data(model)
    _set_shuttle_state(model, data_plain, pos, quat, vel)
    for _ in range(300):
        mujoco.mj_step(model, data_plain)

    qadr, dadr = _shuttle_addresses(model)
    x_aero = float(data_aero.qpos[qadr])
    x_plain = float(data_plain.qpos[qadr])
    assert pos[0] - x_aero < pos[0] - x_plain - 0.4  # drag shortens horizontal travel
    speed_aero = float(np.linalg.norm(data_aero.qvel[dadr : dadr + 3]))
    speed_plain = float(np.linalg.norm(data_plain.qvel[dadr : dadr + 3]))
    assert speed_aero < speed_plain - 2.0


def test_offline_integrator_matches_online(model: mujoco.MjModel) -> None:
    rng = np.random.default_rng(5)
    feed = sample_feed(rng)
    data = _ready_data(model)
    _set_shuttle_state(model, data, feed.launch_pos, launch_quat_from_velocity(feed.launch_vel), feed.launch_vel)
    physics = BadmintonPhysics()
    steps = 500  # 0.5 s at timestep 0.001
    for _ in range(steps):
        physics.substep(model, data)
    qadr, _ = _shuttle_addresses(model)
    online = np.asarray(data.qpos[qadr : qadr + 3], dtype=float)
    offline_traj = integrate_shuttle_flight(
        feed.launch_pos, feed.launch_vel, dt=float(model.opt.timestep), t_max=steps * float(model.opt.timestep)
    )
    offline = offline_traj[-1, 1:4]
    assert float(np.linalg.norm(online - offline)) < 0.15


def test_offline_planned_intercept_matches_online_flight() -> None:
    """Protect the feed scheduler against late-flight aero/integration drift."""

    feed = sample_feed(np.random.default_rng(17))
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    # Isolate free flight: native contacts and stringbed response are impact
    # mechanics, whereas the feed contract describes the pre-impact path.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    data = _ready_data(model)
    _set_shuttle_state(
        model,
        data,
        feed.launch_pos,
        launch_quat_from_velocity(feed.launch_vel),
        feed.launch_vel,
    )
    qadr, _ = _shuttle_addresses(model)
    steps = int(round(feed.intercept_time_s / float(model.opt.timestep)))
    for _ in range(steps):
        data.qfrc_applied[:] = 0.0
        apply_shuttlecock_aero(model, data, default_aero_config())
        mujoco.mj_step(model, data)

    online = np.asarray(data.qpos[qadr : qadr + 3], dtype=float)
    delta = online - feed.intercept_point
    assert float(np.linalg.norm(delta)) < 0.04
    assert abs(float(delta[2])) < 0.01


def _place_shuttle_on_stringbed(
    model: mujoco.MjModel, data: mujoco.MjData, *, closing_speed: float
) -> np.ndarray:
    """Place the cork on the stringbed with a closing normal velocity; returns face normal."""
    racket = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "overall_racket")
    racket_origin = np.asarray(data.xpos[racket], dtype=float)
    racket_rot = np.asarray(data.xmat[racket], dtype=float).reshape(3, 3)
    normal = racket_rot[:, 2]
    stringbed_center = racket_origin + racket_rot @ np.array([0.0, 0.532, 0.0])
    cork_target = stringbed_center + 0.005 * normal  # signed_z=+5mm -> penetration 10mm
    shuttle_pos = cork_target - np.array([0.0, 0.0, 0.011154])  # identity quat cork offset
    _set_shuttle_state(
        model,
        data,
        shuttle_pos,
        np.array([1.0, 0.0, 0.0, 0.0]),
        -closing_speed * normal,
    )
    return normal


def test_event_rebound_flips_velocity(model: mujoco.MjModel) -> None:
    data = _ready_data(model)
    normal = _place_shuttle_on_stringbed(model, data, closing_speed=8.0)
    physics = BadmintonPhysics()
    diag = physics.substep(model, data)
    assert diag["event_rebound_used"] is True
    _, dadr = _shuttle_addresses(model)
    new_vel = np.asarray(data.qvel[dadr : dadr + 3], dtype=float)
    # normal component flips sign; racket is welded to the hand and roughly static
    assert float(np.dot(new_vel, normal)) > 1.0
    assert float(np.dot(new_vel, normal)) < 8.0  # restitution < 1


def test_rebound_cooldown_prevents_retrigger(model: mujoco.MjModel) -> None:
    data = _ready_data(model)
    _place_shuttle_on_stringbed(model, data, closing_speed=8.0)
    physics = BadmintonPhysics(BadmintonPhysicsConfig(rebound_cooldown_substeps=50))
    first = physics.substep(model, data)
    assert first["event_rebound_used"] is True
    assert first["event_stringbed_force_suppressed"] is True
    # re-arm the same closing state; the cooldown must swallow the retrigger
    _place_shuttle_on_stringbed(model, data, closing_speed=8.0)
    second = physics.substep(model, data)
    assert second["event_rebound_used"] is False
    assert second["event_stringbed_force_suppressed"] is True
    assert np.linalg.norm(second["stringbed"]["force_on_shuttle_world"]) > 0.0

    # The force is still measured for diagnostics, but it must not reach the
    # racket chain while the already-resolved event is cooling down.
    racket = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "overall_racket")
    ancestors: set[int] = set()
    body = int(racket)
    while body > 0:
        ancestors.add(body)
        body = int(model.body_parentid[body])
    racket_dofs = np.asarray(
        [int(owner) in ancestors for owner in np.asarray(model.dof_bodyid)],
        dtype=bool,
    )
    np.testing.assert_allclose(data.qfrc_applied[racket_dofs], 0.0, atol=1.0e-12)


def test_qfrc_zeroed_each_substep(model: mujoco.MjModel) -> None:
    data = _ready_data(model)
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_dof = int(model.jnt_dofadr[root_joint])
    data.qfrc_applied[:] = 123.0
    physics = BadmintonPhysics()
    physics.substep(model, data)
    # stale injected forces on the human root must be gone; only shuttle/racket
    # dofs may carry aero/stringbed forces
    assert np.allclose(data.qfrc_applied[root_dof : root_dof + 6], 0.0)
