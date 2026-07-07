#!/usr/bin/env python3
"""Skill pipeline orchestrator: expert -> distill -> hitting base, per action.

Stages (all runnable standalone; ``full-check`` chains a tiny end-to-end pass):

  stage-data      stage local optimized trajectories into the skill cache
  gen-config      emit the per-action expert tracking config
  train-expert    print/launch the mainline tracking command (long-running)
  collect         roll out a tracking expert -> distill shards (per action)
  distill         BC on one or many actions -> frozen base export
                  (multi-action adds a skill one-hot; see train_multi_skill_bc)
  full-check      machinery smoke: collect(tiny) -> distill(tiny) -> load base
                  in the CPU hitting env -> a few env steps

Environment: run from the repo root with the GPU env sourced
(configs/env.sh). The skill cache is exposed to mainline
training via MUSCLEMIMIC_GMR_CACHE_PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")
SKILL_CACHE = REPO_ROOT / "datasets" / "_global" / "muscle_trajectory" / "skill_cache"
DISTILL_ROOT = REPO_ROOT / "datasets" / "_global" / "distill"
PIPELINE_OUT = REPO_ROOT / "outputs" / "skill_pipeline"


def _env_with_cache() -> dict[str, str]:
    env = dict(os.environ)
    env["MUSCLEMIMIC_GMR_CACHE_PATH"] = str(SKILL_CACHE)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("WANDB_MODE", "disabled")
    return env


def _run(cmd: list[str], **kwargs) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), **kwargs)


def stage_data(action: str, **kwargs) -> None:
    _run(
        [
            PYTHON,
            "musclemimic/badminton/skill_pipeline/stage_local_trajectories.py",
            "--action",
            action,
            "--emit-manifest",
            str(REPO_ROOT / "datasets" / action / "manifests" / "skill_pipeline"),
        ],
        env=_env_with_cache(),
    )


def gen_config(action: str, *, num_envs: int, total_timesteps: int) -> None:
    _run(
        [
            PYTHON,
            "musclemimic/badminton/skill_pipeline/generate_expert_config.py",
            "--action",
            action,
            "--num-envs",
            str(num_envs),
            "--total-timesteps",
            str(total_timesteps),
        ]
    )


def train_expert_command(action: str) -> str:
    return (
        f"MUSCLEMIMIC_GMR_CACHE_PATH={SKILL_CACHE} XLA_PYTHON_CLIENT_PREALLOCATE=false "
        f"CUDA_VISIBLE_DEVICES=0 {PYTHON} fullbody/experiment.py "
        f"--config-name=config_specific_task/skill/conf_expert_{action} wandb.mode=disabled"
    )


def collect(
    action: str,
    *,
    teacher_path: str,
    num_envs: int,
    num_steps: int,
    split: str,
    seed: int,
) -> Path:
    out_dir = DISTILL_ROOT / action / split
    manifest_dir = REPO_ROOT / "datasets" / action / "manifests" / "skill_pipeline"
    motion_list = manifest_dir / f"{split}_list.txt"
    motions = [ln.strip() for ln in motion_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
    _run(
        [
            PYTHON,
            "musclemimic/badminton/scripts/collect_forehand_clear_teacher_dataset.py",
            "--teacher-path",
            teacher_path,
            "--output-dir",
            str(out_dir),
            "--num-envs",
            str(num_envs),
            "--num-steps",
            str(num_steps),
            "--seed",
            str(seed),
            "--split",
            split,
            "--motion-path",
            *motions,
        ],
        env=_env_with_cache(),
    )
    return out_dir


def distill(
    actions: list[str],
    *,
    schema_from: str,
    output_dir: Path,
    steps: int,
    hidden: list[int],
) -> Path:
    cmd = [PYTHON, "musclemimic/badminton/skill_pipeline/train_multi_skill_bc.py"]
    for action in actions:
        cmd += ["--dataset", f"{action}={DISTILL_ROOT / action}"]
    cmd += [
        "--schema-from",
        schema_from,
        "--output-dir",
        str(output_dir),
        "--steps",
        str(steps),
        "--hidden",
        *[str(h) for h in hidden],
    ]
    _run(cmd, env=_env_with_cache())
    return output_dir


def full_check(action: str, *, teacher_path: str) -> None:
    """Tiny end-to-end machinery pass; policies are NOT expected to be good."""
    print(f"[1/4] collect tiny distill shards from {teacher_path}")
    collect(action, teacher_path=teacher_path, num_envs=4, num_steps=64, split="train", seed=0)
    collect(action, teacher_path=teacher_path, num_envs=2, num_steps=32, split="val", seed=1)

    print("[2/4] tiny BC distill -> frozen base export")
    base_dir = PIPELINE_OUT / f"base_{action}_smoke"
    distill([action], schema_from=teacher_path, output_dir=base_dir, steps=300, hidden=[64, 64])

    print("[3/4] load base into the CPU hitting env and run steps")
    smoke = f"""
import sys, numpy as np
sys.path.insert(0, {str(REPO_ROOT)!r})
from environment.overall_environment.src.incoming_shuttle_hit_env import IncomingShuttleHitEnv
from environment.overall_environment.src.shuttle_feeder import build_feed_bank
env = IncomingShuttleHitEnv(
    {str(REPO_ROOT / 'environment/overall_environment/assets/overall_incoming_hit_scene.xml')!r},
    feed_bank=build_feed_bank(2, seed=3),
    base_policy_artifact={str(base_dir)!r},
    terminate_on_body_fall=False,
    max_episode_steps=40,
)
obs, info = env.reset(feed_index=0)
for _ in range(10):
    obs, r, term, trunc, info = env.step(np.zeros(env.action_size))
    assert np.isfinite(obs).all() and np.isfinite(r)
print('CPU hitting env with base policy: 10 steps OK, swing_phase=', info['swing_phase'])
"""
    _run([PYTHON, "-c", smoke], env=_env_with_cache())

    print("[4/4] full-check PASSED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("stage-data")
    p.add_argument("--action", required=True)

    p = sub.add_parser("gen-config")
    p.add_argument("--action", required=True)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--total-timesteps", type=int, default=20_480_000)

    p = sub.add_parser("train-expert")
    p.add_argument("--action", required=True)
    p.add_argument("--launch", action="store_true")

    p = sub.add_parser("collect")
    p.add_argument("--action", required=True)
    p.add_argument("--teacher-path", required=True)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=2000)
    p.add_argument("--split", default="train", choices=("train", "val"))
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("distill")
    p.add_argument("--actions", nargs="+", required=True)
    p.add_argument("--schema-from", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 512])

    p = sub.add_parser("full-check")
    p.add_argument("--action", required=True)
    p.add_argument("--teacher-path", required=True)

    args = parser.parse_args()
    if args.stage == "stage-data":
        stage_data(args.action)
    elif args.stage == "gen-config":
        gen_config(args.action, num_envs=args.num_envs, total_timesteps=args.total_timesteps)
    elif args.stage == "train-expert":
        cmd = train_expert_command(args.action)
        print(cmd)
        if args.launch:
            subprocess.run(cmd, shell=True, check=True, cwd=str(REPO_ROOT))
    elif args.stage == "collect":
        collect(
            args.action,
            teacher_path=args.teacher_path,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            split=args.split,
            seed=args.seed,
        )
    elif args.stage == "distill":
        distill(
            args.actions,
            schema_from=args.schema_from,
            output_dir=args.output_dir,
            steps=args.steps,
            hidden=args.hidden,
        )
    elif args.stage == "full-check":
        full_check(args.action, teacher_path=args.teacher_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
