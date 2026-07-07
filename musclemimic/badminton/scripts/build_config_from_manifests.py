#!/usr/bin/env python3
"""Generate the badminton Hydra config from train/val manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_manifest(path: Path) -> list[str]:
    items: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(line.removesuffix(".npz"))
    return items


def _yaml_list(items: list[str], indent: int) -> str:
    spaces = " " * indent
    return "\n".join(f'{spaces}- "{item}"' for item in items)


def _yaml_flow_list(items: list[str]) -> str:
    return "[" + ", ".join(f'"{item}"' for item in items) + "]"


def _yaml_scalar(value: str | None) -> str:
    if value is None:
        return "null"
    return json.dumps(str(value), ensure_ascii=False)


def _hydra_action_run_dir(action: str | None) -> str:
    if action is None:
        return ""
    run_dir = f"datasets/{action}/training/hydra/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}"
    return f"""
hydra:
  run:
    dir: {_yaml_scalar(run_dir)}
"""


def build_config(
    train: list[str],
    val: list[str],
    output: Path,
    num_envs: int,
    total_timesteps: int,
    target_fps: int,
    source_mode: str = "existing_ppo",
    action: str | None = None,
) -> None:
    if not train:
        raise ValueError("train manifest is empty")
    if not val:
        raise ValueError("val manifest is empty")

    tags = ["fullbody", "gmr", "badminton", f"source:{source_mode}"]
    content = f"""# @package _global_
{_hydra_action_run_dir(action)}

defaults:
  - /conf_fullbody_gmr
  - _self_

wandb:
  project: "musclemimic"
  mode: "online"
  tags: {_yaml_flow_list(tags)}

experiment:
  training_action: {_yaml_scalar(action)}

  training_source:
    source_mode: {source_mode}

  env_params:
    env_name: MjxMyoFullBody
    num_envs: {num_envs}
    disable_fingers: true

  total_timesteps: {total_timesteps}

  ppo_config:
    num_steps: 80

  task_factory:
    params:
      amass_dataset_conf:
        dataset_group: null
        rel_dataset_path:
{_yaml_list(train, 10)}
        retargeting_method: gmr
        gmr_config:
          src_human: smplh
          target_fps: {target_fps}
          solver: daqp
          damping: 0.5
          offset_to_ground: false
          use_velocity_limit: false
          use_fitted_shape: true
          shape_fitting_iterations: 500

  validation:
    active: true
    deterministic: false
    num_steps: 500
    num_envs: 20
    num: 20
    amass_dataset_conf:
      dataset_group: null
      rel_dataset_path:
{_yaml_list(val, 8)}
      retargeting_method: gmr
      gmr_config:
        src_human: smplh
        target_fps: {target_fps}
        solver: daqp
        damping: 0.5
        offset_to_ground: false
        use_velocity_limit: false
        use_fitted_shape: true
        shape_fitting_iterations: 500
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)


def main() -> int:
    project_root = _project_root()
    repo_root = _repo_root()
    manifest_root = repo_root / "datasets" / "_global" / "manifests" / "badmintonmimic"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=manifest_root / "train_list.txt")
    parser.add_argument("--val-manifest", type=Path, default=manifest_root / "val_list.txt")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "experiments" / "fullbody" / "config_specific_task" / "conf_fullbody_badminton_gmr.yaml",
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--total-timesteps", type=int, default=20480000)
    parser.add_argument("--fps", "--target-fps", dest="target_fps", type=int, default=30)
    parser.add_argument("--source-mode", default="existing_ppo")
    parser.add_argument(
        "--action",
        default=None,
        help="Dataset action name. When set, training artifacts route to datasets/<action>/training.",
    )
    args = parser.parse_args()

    train = _read_manifest(args.train_manifest)
    val = _read_manifest(args.val_manifest)
    build_config(
        train,
        val,
        args.output,
        args.num_envs,
        args.total_timesteps,
        args.target_fps,
        source_mode=args.source_mode,
        action=args.action,
    )
    print(f"[OK] Wrote {args.output}")
    print(f"     train_motions={len(train)} val_motions={len(val)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
