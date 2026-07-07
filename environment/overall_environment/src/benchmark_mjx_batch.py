"""GPU batch benchmark for the MJX badminton substep pipeline.

Vmaps the full substep (aero + stringbed + event rebound + mjx.step) over a
batch of environments, reports throughput, and checks batch-wide finiteness
plus hand-racket weld integrity on the accelerator.

Run from the repository root (LD_LIBRARY_PATH must not contain the system
CUDA toolkit; source configs/env.sh first):

    .venv/bin/python -m environment.overall_environment.src.benchmark_mjx_batch --num-envs 1024 --steps 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from environment.overall_environment.src.badminton_physics_mjx import (
    make_ids,
    make_params,
    make_substep_fn,
)
from environment.overall_environment.src.paths import default_incoming_scene_path
from environment.overall_environment.src.shuttle_feeder import (
    build_feed_bank,
    launch_quat_from_velocity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--xml", type=Path, default=default_incoming_scene_path())
    args = parser.parse_args()

    print("jax devices:", jax.devices())
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    ids = make_ids(model)
    params = make_params(model)

    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    mx = mjx.put_model(model)
    dx = mjx.put_data(model, data)

    # batch: same ready pose, per-env feed launch states
    feeds = build_feed_bank(min(args.num_envs, 64), seed=21)
    qadr = int(model.jnt_qposadr[model.body_jntadr[ids.shuttle_body]])
    dadr = ids.shuttle_dofadr
    launch_qpos = np.stack(
        [
            np.concatenate(
                [
                    feeds[i % len(feeds)].launch_pos,
                    launch_quat_from_velocity(feeds[i % len(feeds)].launch_vel),
                ]
            )
            for i in range(args.num_envs)
        ]
    )
    launch_qvel = np.stack(
        [feeds[i % len(feeds)].launch_vel for i in range(args.num_envs)]
    )

    def batchify(d):
        qpos = jnp.tile(d.qpos[None], (args.num_envs, 1))
        qvel = jnp.tile(d.qvel[None], (args.num_envs, 1))
        qpos = qpos.at[:, qadr : qadr + 7].set(jnp.asarray(launch_qpos))
        qvel = qvel.at[:, dadr : dadr + 3].set(jnp.asarray(launch_qvel))
        batch = jax.tree_util.tree_map(
            lambda x: jnp.tile(x[None], (args.num_envs,) + (1,) * x.ndim)
            if hasattr(x, "ndim")
            else x,
            d,
        )
        return batch.replace(qpos=qpos, qvel=qvel)

    batch = jax.jit(batchify)(dx)
    forward_batched = jax.jit(jax.vmap(lambda d: mjx.forward(mx, d)))
    batch = forward_batched(batch)

    substep = make_substep_fn(mx, ids, params)

    def rollout(d, cooldown, steps):
        def body(carry, _):
            d, cd = carry
            d, cd, _diag = substep(d, cd)
            return (d, cd), None

        (d, cd), _ = jax.lax.scan(body, (d, cooldown), None, length=steps)
        return d, cd

    rollout_batched = jax.jit(
        jax.vmap(rollout, in_axes=(0, 0, None)), static_argnums=(2,)
    )
    cooldown0 = jnp.zeros((args.num_envs,), dtype=jnp.int32)

    t0 = time.time()
    out, _cd = rollout_batched(batch, cooldown0, args.steps)
    out.qpos.block_until_ready()
    compile_and_run_s = time.time() - t0

    t1 = time.time()
    out2, _cd = rollout_batched(batch, cooldown0, args.steps)
    out2.qpos.block_until_ready()
    steady_s = time.time() - t1

    total_substeps = args.num_envs * args.steps
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_palm_grip_site")
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "overall_grip_pose_site")
    weld_dist = np.asarray(
        jnp.linalg.norm(out2.site_xpos[:, palm] - out2.site_xpos[:, grip], axis=-1)
    )
    report = {
        "num_envs": args.num_envs,
        "steps": args.steps,
        "compile_plus_first_run_s": round(compile_and_run_s, 1),
        "steady_state_s": round(steady_s, 3),
        "substeps_per_second": round(total_substeps / steady_s),
        "policy_steps_per_second_at_10_substeps": round(total_substeps / steady_s / 10),
        "all_finite": bool(jnp.isfinite(out2.qpos).all()),
        "weld_palm_grip_dist_mean_m": float(weld_dist.mean()),
        "weld_palm_grip_dist_max_m": float(weld_dist.max()),
        "device": str(jax.devices()[0]),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
