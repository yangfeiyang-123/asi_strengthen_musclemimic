"""Feasibility probe: badminton substep pipeline on the MJX Warp backend.

Warp batching semantics: contact pools/counters are shared across worlds
(``naconmax`` is the total budget); every other Data field carries a leading
world dimension and warp kernels vmap over it implicitly — ``mjx.step`` is
called directly on the semi-batched Data (no ``jax.vmap``).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco.mjx.warp import types as mjxw_types

from environment.overall_environment.src.badminton_physics_mjx import (
    aero_force_torque,
    event_rebound_velocity,
    make_ids,
    make_params,
    stringbed_contact,
)
from environment.overall_environment.src.paths import default_incoming_scene_path

NUM_ENVS = int(sys.argv[1]) if len(sys.argv) > 1 else 256
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
NCONMAX = 128  # per-env contact budget

print("devices:", jax.devices())
model = mujoco.MjModel.from_xml_path(str(default_incoming_scene_path()))
ids = make_ids(model)
params = make_params(model)
data = mujoco.MjData(model)
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

mx = mjx.put_model(model, impl="warp")
dx = mjx.make_data(
    model,
    impl="warp",
    nconmax=NCONMAX * NUM_ENVS,
    njmax=NCONMAX * 4,
    naconmax=NCONMAX * NUM_ENVS,
)
dx = dx.replace(
    qpos=jnp.asarray(data.qpos), qvel=jnp.asarray(data.qvel), act=jnp.asarray(data.act)
)

_UNBATCHED = frozenset(k for k, v in mjxw_types._BATCH_DIM["Data"].items() if not v)


def _attr_name(path) -> str:
    return "__".join(p.name for p in path if hasattr(p, "name")).removeprefix("_impl__")


def batchify(d, num_envs: int):
    def fn(path, x):
        if _attr_name(path) in _UNBATCHED or not hasattr(x, "ndim"):
            return x
        return jnp.tile(x[None], (num_envs,) + (1,) * x.ndim)

    return jax.tree_util.tree_map_with_path(fn, d)


batch = jax.jit(batchify, static_argnums=(1,))(dx, NUM_ENVS)
print("== batched forward (implicit warp vmap) ==")
batch = jax.jit(lambda d: mjx.forward(mx, d))(batch)
print("forward ok; qpos shape:", batch.qpos.shape, "cork z:", float(batch.site_xpos[0, ids.cork_site, 2]))

# --- batched badminton substep: pure-jax forces vmapped over world dim -----
dof_mask_shuttle = jnp.asarray(np.asarray(model.dof_bodyid) == ids.shuttle_body)
dof_mask_racket = jnp.asarray(np.asarray(model.dof_bodyid) == ids.racket_body)


def _qfrc_from_point_force(cdof, subtree_com_root, dof_mask, force, torque, point):
    offset = point - subtree_com_root
    jacp = cdof[:, 3:] + jnp.cross(cdof[:, :3], jnp.broadcast_to(offset, cdof[:, :3].shape))
    jacr = cdof[:, :3]
    return dof_mask * (jacp @ force + jacr @ torque)


def _point_velocity(cvel, subtree_com_root, point):
    return cvel[3:] + jnp.cross(cvel[:3], point - subtree_com_root)


def substep_batched(d, cooldown):
    shuttle_com = d.xipos[:, ids.shuttle_body]
    shuttle_rot = d.xmat[:, ids.shuttle_body].reshape(-1, 3, 3)
    nose_axis = shuttle_rot[:, :, 2]
    omega = d.cvel[:, ids.shuttle_body, :3]
    com_root = d.subtree_com[:, ids.shuttle_root]
    racket_root_com = d.subtree_com[:, ids.racket_root]
    v_com = jax.vmap(_point_velocity)(d.cvel[:, ids.shuttle_body], com_root, shuttle_com)

    aero_f, aero_t = jax.vmap(
        lambda v, o, n: aero_force_torque(params, v_world=v, omega_world=o, nose_axis_world=n)
    )(v_com, omega, nose_axis)

    contact_point = d.site_xpos[:, ids.cork_site]
    v_shuttle_pt = jax.vmap(_point_velocity)(d.cvel[:, ids.shuttle_body], com_root, contact_point)
    v_racket_pt = jax.vmap(_point_velocity)(d.cvel[:, ids.racket_body], racket_root_com, contact_point)
    contact = jax.vmap(
        lambda ro, rr, cp, vs, vr: stringbed_contact(
            params,
            racket_origin=ro,
            racket_rot=rr,
            contact_point=cp,
            v_shuttle_point=vs,
            v_racket_point=vr,
        )
    )(d.xpos[:, ids.racket_body], d.xmat[:, ids.racket_body].reshape(-1, 3, 3), contact_point, v_shuttle_pt, v_racket_pt)

    cdof = d._impl.cdof
    qfrc_aero = jax.vmap(
        lambda c, s, f, t, p: _qfrc_from_point_force(c, s, dof_mask_shuttle, f, t, p)
    )(cdof, com_root, aero_f, aero_t, shuttle_com)
    zero3 = jnp.zeros_like(aero_f)
    qfrc_bed = jax.vmap(
        lambda c, s, f, t, p: _qfrc_from_point_force(c, s, dof_mask_shuttle, f, t, p)
    )(cdof, com_root, contact["force_on_shuttle"], zero3, contact_point) + jax.vmap(
        lambda c, s, f, t, p: _qfrc_from_point_force(c, s, dof_mask_racket, f, t, p)
    )(cdof, racket_root_com, -contact["force_on_shuttle"], zero3, contact_point)

    trigger = (
        (cooldown <= 0)
        & contact["active"]
        & (contact["relative_normal_velocity"] < -params.min_speed_for_event)
    )
    dadr = ids.shuttle_dofadr
    new_vel = jax.vmap(
        lambda sv, rv, n: event_rebound_velocity(
            params, shuttle_velocity=sv, racket_surface_velocity=rv, normal_world=n
        )
    )(d.qvel[:, dadr : dadr + 3], v_racket_pt, contact["normal_world"])
    qvel = jnp.where(
        trigger[:, None],
        d.qvel.at[:, dadr : dadr + 3].set(new_vel),
        d.qvel,
    )
    qfrc = jnp.where(trigger[:, None], qfrc_aero, qfrc_aero + qfrc_bed)
    cooldown = jnp.where(
        trigger, jnp.asarray(params.rebound_cooldown_substeps, cooldown.dtype), jnp.maximum(cooldown - 1, 0)
    )
    d = d.replace(qvel=qvel, qfrc_applied=qfrc)
    d = mjx.step(mx, d)
    return d, cooldown


def rollout(d, cooldown, steps):
    def body(carry, _):
        d, cd = carry
        return substep_batched(d, cd), None

    (d, cd), _ = jax.lax.scan(body, (d, cooldown), None, length=steps)
    return d, cd


rollout_jit = jax.jit(rollout, static_argnums=(2,))
cooldown0 = jnp.zeros((NUM_ENVS,), dtype=jnp.int32)

print(f"== batched substep benchmark N={NUM_ENVS} x {STEPS} ==")
t0 = time.time()
out, _ = rollout_jit(batch, cooldown0, STEPS)
out.qpos.block_until_ready()
print(f"compile+first: {time.time()-t0:.1f}s")
t1 = time.time()
out, _ = rollout_jit(batch, cooldown0, STEPS)
out.qpos.block_until_ready()
steady = time.time() - t1
total = NUM_ENVS * STEPS
print(
    f"steady: {steady:.2f}s  {total/steady:,.0f} substeps/s  "
    f"{total/steady/10:,.0f} policy-steps/s @10 substeps"
)
print("all finite:", bool(jnp.isfinite(out.qpos).all()))
