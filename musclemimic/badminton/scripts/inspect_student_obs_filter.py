"""Inspect ForehandClear student observation filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from loco_mujoco.task_factories import TaskFactory
from musclemimic.distill.obs_filter import build_student_obs_indices
from musclemimic.runner.eval_utils import apply_temporal_params


def _load_config(config_name: str):
    fullbody_dir = Path(__file__).resolve().parents[3] / "fullbody"
    with initialize_config_dir(version_base=None, config_dir=str(fullbody_dir)):
        return compose(config_name=config_name.removesuffix(".yaml"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect student_obs_filter dimensions and dropped goal lookahead.")
    parser.add_argument("--config-name", default="config_specific_task/conf_fullbody_badminton_student_gmr")
    parser.add_argument("--motion-path", nargs="+", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    config = _load_config(args.config_name)
    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"] = True
    config.experiment.env_params["num_envs"] = 1
    if args.motion_path:
        config.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path = list(args.motion_path)
    apply_temporal_params(config)

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(
        **OmegaConf.to_container(config.experiment.env_params, resolve=True),
        **OmegaConf.to_container(config.experiment.task_factory.params, resolve=True),
    )
    spec = build_student_obs_indices(env, config.experiment.student_obs_filter)
    kept_goal_indices = [
        int(idx)
        for idx in spec.student_indices
        if int(idx) in set(int(goal_idx) for goal_idx in spec.goal_indices)
    ]
    payload = {
        "raw_obs_dim": int(spec.raw_obs_dim),
        "goal_dim": int(spec.goal_indices.size),
        "state_dim": int(spec.state_indices.size),
        "student_obs_dim": int(spec.student_obs_dim),
        "phase_index_raw": None if spec.phase_index is None else int(spec.phase_index),
        "phase_index_student": spec.phase_student_index,
        "kept_goal_indices": kept_goal_indices,
        "dropped_goal_dim": int(spec.goal_indices.size - len(kept_goal_indices)),
        "student_obs_filter": OmegaConf.to_container(config.experiment.student_obs_filter, resolve=True),
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
