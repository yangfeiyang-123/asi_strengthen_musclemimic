from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.experimental import enable_x64  # noqa: E402

from environment.overall_environment.src.badminton_physics import (  # noqa: E402
    BadmintonPhysics,
    BadmintonPhysicsConfig,
)
from environment.overall_environment.src.badminton_physics_mjx import (  # noqa: E402
    aero_force_torque,
    body_dof_mask,
    event_reaction_impulses,
    event_rebound_velocity,
    make_ids,
    make_params,
    make_substep_fn,
    stringbed_contact,
)
from environment.overall_environment.src.paths import default_incoming_scene_path  # noqa: E402
from environment.overall_environment.src.shuttle_feeder import (  # noqa: E402
    launch_quat_from_velocity,
    sample_feed,
)
from environment.racket.src.racket_stringbed import (  # noqa: E402
    RacketGeometry,
    StringbedParams,
    local_impact_coordinates,
    stringbed_stiffness_at,
)
from environment.shuttlecock.src.shuttlecock_aero import (  # noqa: E402
    ShuttlecockAeroConfig,
    compute_shuttlecock_aero,
)
from environment.shuttlecock.src.shuttlecock_racket_impact import (  # noqa: E402
    ShuttlecockImpactConfig,
    compute_equal_opposite_event_impulses,
    compute_event_rebound,
)

SCENE_XML = default_incoming_scene_path()
RUN_MJX_STACK = os.environ.get("RUN_MJX_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not SCENE_XML.is_file(),
    reason="incoming scene XML not built; run environment.overall_environment.src.incoming_scene",
)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE_XML))


@pytest.fixture(scope="module")
def params(model):
    return make_params(model)


def test_aero_parity_with_numpy(model, params) -> None:
    with enable_x64():
        cfg = ShuttlecockAeroConfig()
        rng = np.random.default_rng(0)
        for _ in range(25):
            v = rng.normal(0, 15, 3)
            omega = rng.normal(0, 30, 3)
            nose = rng.normal(0, 1, 3)
            nose /= np.linalg.norm(nose)
            com = rng.normal(0, 2, 3)
            force_np, torque_np, _cp, _diag = compute_shuttlecock_aero(
                mass_kg=params.shuttle_mass_kg,
                gravity=np.array([0.0, 0.0, -9.81]),
                wind=np.zeros(3),
                v_world=v,
                omega_world=omega,
                nose_axis_world=nose,
                com_world=com,
                cfg=cfg,
            )
            force_jx, torque_jx = aero_force_torque(
                params,
                v_world=jnp.asarray(v),
                omega_world=jnp.asarray(omega),
                nose_axis_world=jnp.asarray(nose),
            )
            np.testing.assert_allclose(np.asarray(force_jx), force_np, atol=1e-8)
            np.testing.assert_allclose(np.asarray(torque_jx), torque_np, atol=1e-8)


def test_stringbed_parity_with_numpy(params) -> None:
    with enable_x64():
        geom = RacketGeometry()
        sp = StringbedParams()
        rng = np.random.default_rng(1)
        active_seen = 0
        for _ in range(60):
            racket_origin = rng.normal(0, 1, 3)
            quat = rng.normal(0, 1, 4)
            quat /= np.linalg.norm(quat)
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, quat)
            rot = rot.reshape(3, 3)
            # sample near the stringbed so a good share of draws are active
            local = np.array(
                [
                    rng.uniform(-geom.stringbed_half_width, geom.stringbed_half_width),
                    geom.stringbed_center_y + rng.uniform(-geom.stringbed_half_length, geom.stringbed_half_length),
                    rng.uniform(-0.02, 0.02),
                ]
            )
            contact_point = racket_origin + rot @ local
            v_shuttle = rng.normal(0, 6, 3)
            v_racket = rng.normal(0, 3, 3)

            # numpy reference (same math as apply_stringbed_force without mujoco)
            p_local, rho2 = local_impact_coordinates(racket_origin, rot, contact_point, geom)
            signed_z = float(p_local[2])
            penetration = sp.cork_radius + geom.stringbed_proxy_thickness - abs(signed_z)
            if rho2 > 1.0 or penetration <= 0.0:
                expected = np.zeros(3)
            else:
                side = 1.0 if signed_z >= 0.0 else -1.0
                normal = side * rot[:, 2]
                v_rel = v_shuttle - v_racket
                v_n = float(np.dot(v_rel, normal))
                v_t = v_rel - v_n * normal
                f_n = max(0.0, stringbed_stiffness_at(rho2, sp) * penetration - sp.normal_damping * v_n)
                tangential = -sp.tangential_damping * v_t
                tn, tl = float(np.linalg.norm(tangential)), sp.tangential_mu * f_n
                if tn > tl > 0.0:
                    tangential *= tl / tn
                expected = f_n * normal + tangential
                active_seen += 1

            result = stringbed_contact(
                params,
                racket_origin=jnp.asarray(racket_origin),
                racket_rot=jnp.asarray(rot),
                contact_point=jnp.asarray(contact_point),
                v_shuttle_point=jnp.asarray(v_shuttle),
                v_racket_point=jnp.asarray(v_racket),
            )
            np.testing.assert_allclose(
                np.asarray(result["force_on_shuttle"]), expected, atol=1e-6
            )
    assert active_seen >= 10


def test_event_rebound_parity_with_numpy(params) -> None:
    with enable_x64():
        cfg = ShuttlecockImpactConfig()
        rng = np.random.default_rng(2)
        for _ in range(30):
            v_shuttle = rng.normal(0, 12, 3)
            v_racket = rng.normal(0, 8, 3)
            normal = rng.normal(0, 1, 3)
            normal /= np.linalg.norm(normal)
            expected, _diag = compute_event_rebound(
                shuttle_velocity_world=v_shuttle,
                racket_surface_velocity_world=v_racket,
                normal_world=normal,
                cfg=cfg,
            )
            result = event_rebound_velocity(
                params,
                shuttle_velocity=jnp.asarray(v_shuttle),
                racket_surface_velocity=jnp.asarray(v_racket),
                normal_world=jnp.asarray(normal),
            )
            np.testing.assert_allclose(np.asarray(result), expected, atol=1e-8)


def test_event_reaction_impulse_parity_and_momentum_balance(params) -> None:
    with enable_x64():
        before = np.asarray([-8.0, 1.5, -0.25])
        after = np.asarray([4.0, 1.275, -0.2125])
        expected_shuttle, expected_racket = compute_equal_opposite_event_impulses(
            shuttle_mass_kg=params.shuttle_mass_kg,
            velocity_before_world=before,
            velocity_after_world=after,
        )
        actual_shuttle, actual_racket = event_reaction_impulses(
            params,
            shuttle_velocity_before=jnp.asarray(before),
            shuttle_velocity_after=jnp.asarray(after),
        )
        np.testing.assert_allclose(np.asarray(actual_shuttle), expected_shuttle, atol=1e-12)
        np.testing.assert_allclose(np.asarray(actual_racket), expected_racket, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(actual_shuttle + actual_racket), np.zeros(3), atol=1e-12
        )


def test_event_reaction_reaches_exact_child_ancestors_cpu_mjx_consistently() -> None:
    """Exercise the real CPU/MJX substeps on a small jointless-racket model."""

    xml = """
    <mujoco>
      <option timestep=".001" gravity="0 0 0"/>
      <worldbody>
        <body name="human">
          <freejoint name="root"/>
          <geom type="sphere" size=".05" mass="1" contype="0" conaffinity="0"/>
          <body name="overall_racket">
            <geom type="box" size=".05 .6 .005" mass=".09" contype="0" conaffinity="0"/>
          </body>
        </body>
        <body name="overall_shuttle" pos="0 .532 .005">
          <freejoint name="overall_shuttle_free"/>
          <geom type="sphere" size=".01" mass=".005" contype="0" conaffinity="0"/>
          <site name="overall_cork_contact_site"/>
        </body>
      </worldbody>
    </mujoco>
    """
    small_model = mujoco.MjModel.from_xml_string(xml)
    ids = make_ids(small_model)
    params = make_params(small_model)
    initial = mujoco.MjData(small_model)
    initial.qvel[ids.shuttle_dofadr + 2] = -8.0
    mujoco.mj_forward(small_model, initial)

    cpu_data = mujoco.MjData(small_model)
    cpu_data.qpos[:] = initial.qpos
    cpu_data.qvel[:] = initial.qvel
    mujoco.mj_forward(small_model, cpu_data)
    cpu_diag = BadmintonPhysics().substep(small_model, cpu_data)
    assert cpu_diag["event_rebound_used"] is True

    from mujoco import mjx

    mx = mjx.put_model(small_model)
    dx = mjx.forward(mx, mjx.put_data(small_model, initial))
    dx, _cooldown, mjx_diag = make_substep_fn(mx, ids, params)(dx, jnp.asarray(0))
    assert bool(np.asarray(mjx_diag["event_rebound_used"])) is True
    np.testing.assert_allclose(
        np.asarray(
            mjx_diag["event_impulse_on_shuttle_world_ns"]
            + mjx_diag["event_impulse_on_racket_world_ns"]
        ),
        0.0,
        atol=1e-8,
    )
    # Root translation and rotation both react because the impulse is applied
    # at the off-COM cork point on a jointless child racket.
    assert np.linalg.norm(np.asarray(dx.qvel[:6])) > 0.0
    np.testing.assert_allclose(np.asarray(dx.qvel), cpu_data.qvel, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not RUN_MJX_STACK, reason="set RUN_MJX_TESTS=1 to run the mjx full-stack test")
def test_flight_trajectory_parity_cpu_vs_mjx(model, params) -> None:
    from mujoco import mjx

    ids = make_ids(model)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    feed = sample_feed(np.random.default_rng(7))
    qadr, dadr = int(model.jnt_qposadr[model.body_jntadr[ids.shuttle_body]]), ids.shuttle_dofadr
    data.qpos[qadr : qadr + 3] = feed.launch_pos
    data.qpos[qadr + 3 : qadr + 7] = launch_quat_from_velocity(feed.launch_vel)
    data.qvel[dadr : dadr + 3] = feed.launch_vel
    mujoco.mj_forward(model, data)

    steps = 150
    # CPU reference
    cpu_data = mujoco.MjData(model)
    cpu_data.qpos[:] = data.qpos
    cpu_data.qvel[:] = data.qvel
    mujoco.mj_forward(model, cpu_data)
    physics = BadmintonPhysics(BadmintonPhysicsConfig())
    for _ in range(steps):
        physics.substep(model, cpu_data)
    cpu_shuttle = np.array(cpu_data.qpos[qadr : qadr + 3])

    # MJX rollout
    mx = mjx.put_model(model)
    dx = mjx.put_data(model, data)
    dx = mjx.forward(mx, dx)
    substep = make_substep_fn(mx, ids, params)

    def body(carry, _):
        d, cooldown = carry
        d, cooldown, _diag = substep(d, cooldown)
        return (d, cooldown), None

    (dx, _cd), _ = jax.lax.scan(body, (dx, jnp.asarray(0)), None, length=steps)
    mjx_shuttle = np.asarray(dx.qpos[qadr : qadr + 3])

    assert np.linalg.norm(mjx_shuttle - cpu_shuttle) < 0.05


def test_batched_qfrc_formula_matches_mj_applyFT(model) -> None:
    """The cdof-based force mapping used by the batched (warp) substep."""
    ids = make_ids(model)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    rng = np.random.default_rng(0)
    data.qvel[:] = rng.normal(0, 0.5, model.nv)
    mujoco.mj_forward(model, data)

    for body_id, root_id in (
        (ids.shuttle_body, ids.shuttle_root),
        (ids.racket_body, ids.racket_root),
    ):
        force = rng.normal(0, 3, 3)
        torque = rng.normal(0, 0.05, 3)
        point = np.array(data.xipos[body_id]) + rng.normal(0, 0.1, 3)
        reference = np.zeros(model.nv)
        mujoco.mj_applyFT(model, data, force, torque, point, body_id, reference)

        dof_mask = body_dof_mask(model, body_id).astype(float)
        cdof = np.array(data.cdof)
        offset = point - np.array(data.subtree_com[root_id])
        jacp = cdof[:, 3:] + np.cross(cdof[:, :3], np.broadcast_to(offset, (model.nv, 3)))
        mine = dof_mask * (jacp @ force + cdof[:, :3] @ torque)
        np.testing.assert_allclose(mine, reference, atol=1e-12)


def test_body_dof_mask_propagates_jointless_child_force_to_ancestor_chain() -> None:
    xml = """
    <mujoco>
      <worldbody>
        <body name="root">
          <freejoint/>
          <geom type="sphere" size=".02" mass="1"/>
          <body name="arm"><joint name="hinge" axis="0 0 1"/>
            <geom type="capsule" size=".01" fromto="0 0 0 0 .5 0" mass=".2"/>
            <body name="racket" pos="0 1 0"><geom type="sphere" size=".01" mass=".1"/></body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    chain_model = mujoco.MjModel.from_xml_string(xml)
    racket_id = mujoco.mj_name2id(
        chain_model,
        mujoco.mjtObj.mjOBJ_BODY,
        "racket",
    )
    mask = body_dof_mask(chain_model, racket_id)
    assert mask.shape == (chain_model.nv,)
    assert int(mask.sum()) == 7  # six root DOFs plus the arm hinge
    assert not np.any(np.asarray(chain_model.dof_bodyid)[mask] == racket_id)

    data = mujoco.MjData(chain_model)
    mujoco.mj_forward(chain_model, data)
    point = np.asarray(data.xpos[racket_id], dtype=float)
    force = np.asarray([1.0, 2.0, 3.0])
    reference = np.zeros(chain_model.nv)
    mujoco.mj_applyFT(
        chain_model,
        data,
        force,
        np.zeros(3),
        point,
        racket_id,
        reference,
    )
    assert np.any(np.abs(reference[mask]) > 0.0)
    assert not np.any(np.abs(reference[~mask]) > 0.0)
