"""Headless rollout demo for the two-player rally environment.

Runs one episode and optionally renders a video.  Without trained policies the
players have no muscle tone, so ``--freeze-humans`` (default on for video)
kinematically holds both players in the ready pose while the serve flies its
full drag-shaped arc across the court -- a scene/aero smoke, not a policy demo.

Usage:
    MUJOCO_GL=egl .venv/bin/python -m environment.double_play.src.run_double_play_demo \
        --video outputs/double_play_demo.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from environment.double_play.src.double_play_env import DoublePlayRallyEnv


def _freeze_humans(env: DoublePlayRallyEnv, ready_qpos: np.ndarray) -> None:
    """Clamp both players back to the ready pose, leaving the shuttle free."""
    qadr = env._shuttle_qadr
    dadr = env._shuttle_dadr
    shuttle_qpos = env.data.qpos[qadr : qadr + 7].copy()
    shuttle_qvel = env.data.qvel[dadr : dadr + 6].copy()
    env.data.qpos[:] = ready_qpos
    env.data.qvel[:] = 0.0
    env.data.qpos[qadr : qadr + 7] = shuttle_qpos
    env.data.qvel[dadr : dadr + 6] = shuttle_qvel
    mujoco.mj_forward(env.model, env.data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=None, help="Optional output mp4 path.")
    parser.add_argument("--steps", type=int, default=250, help="Control steps to roll out.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--serve-receiver", choices=["random", "p1", "p2"], default="p1"
    )
    parser.add_argument(
        "--freeze-humans",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hold both players in the ready pose (scene/aero smoke without policies).",
    )
    parser.add_argument(
        "--terminate-on-fall",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="End the episode when a body falls (off by default: untrained bodies always fall).",
    )
    parser.add_argument(
        "--extra-seconds",
        type=float,
        default=2.0,
        help="Keep integrating and rendering pure physics after the rally episode ends.",
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    env = DoublePlayRallyEnv(seed=args.seed, terminate_on_body_fall=args.terminate_on_fall)
    obs, info = env.reset(serve_receiver=args.serve_receiver)
    ready_qpos = np.asarray(env.model.key_qpos[env.keyframe_id], dtype=float).copy()
    print(
        f"serve to {info['serve_receiver']}: shuttle at {np.round(info['shuttle_pos'], 2)}, "
        f"obs size {env.observation_size}, action size {env.action_size} per player"
    )

    renderer = None
    frames = []
    if args.video is not None:
        env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, args.width)
        env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, args.height)
        renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)

    zero = {name: np.zeros(env.action_size) for name in ("p1", "p2")}
    frame_every = max(1, int(round(1.0 / (args.fps * env.control_substeps * env.model.opt.timestep))))

    def capture(step: int) -> None:
        if renderer is not None and step % frame_every == 0:
            renderer.update_scene(env.data, camera="overall_view")
            frames.append(renderer.render())

    step = 0
    for step in range(args.steps):
        obs, rewards, terminated, truncated, info = env.step(zero)
        if args.freeze_humans:
            _freeze_humans(env, ready_qpos)
        capture(step)
        if terminated or truncated:
            print(
                f"episode ended at step {step}: reason={info['termination_reason']}, "
                f"shuttle={np.round(info['shuttle_pos'], 2)}"
            )
            break
    else:
        print(f"rolled {args.steps} steps without termination")

    extra_steps = int(round(args.extra_seconds / (env.control_substeps * env.model.opt.timestep)))
    for extra in range(extra_steps):
        for _ in range(env.control_substeps):
            env.physics.substep(env.model, env.data)
        if args.freeze_humans:
            _freeze_humans(env, ready_qpos)
        capture(step + 1 + extra)
    if extra_steps:
        print(f"rendered {args.extra_seconds:.1f}s of post-episode physics")

    if renderer is not None:
        import imageio.v2 as imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, frames, fps=args.fps)
        print(f"wrote {len(frames)} frames to {args.video}")
        renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
