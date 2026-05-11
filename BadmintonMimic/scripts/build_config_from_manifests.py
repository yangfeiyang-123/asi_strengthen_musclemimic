#!/usr/bin/env python3
"""Generate the BadmintonMimic Hydra config from train/val manifests."""

from __future__ import annotations

import argparse
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def build_config(train: list[str], val: list[str], output: Path, num_envs: int, total_timesteps: int) -> None:
    if not train:
        raise ValueError("train manifest is empty")
    if not val:
        raise ValueError("val manifest is empty")

    content = f"""# @package _global_

defaults:
  - /conf_fullbody_gmr
  - _self_

wandb:
  project: "musclemimic"
  mode: "online"
  tags: ["fullbody", "gmr", "badminton"]

experiment:
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
          target_fps: 30
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
        target_fps: 30
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=project_root / "manifests" / "train_list.txt")
    parser.add_argument("--val-manifest", type=Path, default=project_root / "manifests" / "val_list.txt")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "experiments" / "fullbody" / "config_specific_task" / "conf_fullbody_badminton_gmr.yaml",
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--total-timesteps", type=int, default=20480000)
    args = parser.parse_args()

    train = _read_manifest(args.train_manifest)
    val = _read_manifest(args.val_manifest)
    build_config(train, val, args.output, args.num_envs, args.total_timesteps)
    print(f"[OK] Wrote {args.output}")
    print(f"     train_motions={len(train)} val_motions={len(val)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
