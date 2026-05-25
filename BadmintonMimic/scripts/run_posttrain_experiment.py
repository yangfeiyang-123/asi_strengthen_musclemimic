#!/usr/bin/env python3
"""Prepare and run reusable badminton PostTrain experiments."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL = ("experiment_id", "action", "output_root", "resume_from", "reference", "arms")
REQUIRED_REFERENCE = ("train", "validation")
DEFAULT_BASE_REWARD = {
    "qpos_w_sum": 0.05,
    "qvel_w_sum": 0.08,
    "root_pos_w_sum": 0.35,
    "root_vel_w_sum": 0.25,
    "rpos_w_sum": 0.30,
    "rquat_w_sum": 0.01,
    "rvel_w_sum": 0.06,
    "absolute_site_reward_sites": ["right_hand_mimic"],
    "absolute_site_w_sum": 0.10,
    "absolute_site_w_exp": 10.0,
}
DEFAULT_TERMINAL = {
    "mean_site_deviation_threshold": 0.45,
    "root_deviation_threshold": 0.30,
    "root_orientation_threshold": 0.70,
    "enable_site_check": True,
}
DEFAULT_ENV = {
    "env_name": "MjxMyoFullBody",
    "disable_fingers": True,
    "terminal_state_type": "MeanRelativeSiteDeviationWithRootTerminalStateHandler",
}


@dataclass(frozen=True)
class GeneratedConfig:
    arm_id: str
    config_name: str
    output_copy: Path
    hydra_config: Path


@dataclass(frozen=True)
class PrepareResult:
    output_dir: Path
    report_path: Path
    generated_configs: dict[str, GeneratedConfig]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base or _project_root()) / path


def _normalize_motion_path(value: str) -> str:
    return value.removesuffix(".npz")


def _normalize_motion_list(items: list[str]) -> list[str]:
    return [_normalize_motion_path(str(item)) for item in items]


def _require_mapping(mapping: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{context} missing required keys: {', '.join(missing)}")


def load_spec(path: str | Path) -> dict[str, Any]:
    """Load and validate a PostTrain experiment spec."""
    spec_path = Path(path)
    with spec_path.open() as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"{spec_path} must contain a YAML mapping")

    _require_mapping(spec, REQUIRED_TOP_LEVEL, "spec")
    if not isinstance(spec["reference"], dict):
        raise ValueError("spec.reference must be a mapping")
    _require_mapping(spec["reference"], REQUIRED_REFERENCE, "spec.reference")
    if not isinstance(spec["arms"], list) or not spec["arms"]:
        raise ValueError("spec.arms must be a non-empty list")

    reference = dict(spec["reference"])
    for key in ("train", "validation", "stress_test", "excluded"):
        reference[key] = _normalize_motion_list(reference.get(key, []))
    spec["reference"] = reference

    normalized_arms = []
    seen_ids: set[str] = set()
    for arm in spec["arms"]:
        if not isinstance(arm, dict) or "id" not in arm:
            raise ValueError("each arm must be a mapping with an id")
        arm = dict(arm)
        arm_id = str(arm["id"])
        if arm_id in seen_ids:
            raise ValueError(f"duplicate arm id: {arm_id}")
        seen_ids.add(arm_id)
        arm["id"] = arm_id
        normalized_arms.append(arm)
    spec["arms"] = normalized_arms
    spec["_spec_path"] = str(spec_path)
    return spec


def _output_dir(spec: dict[str, Any]) -> Path:
    return _as_path(spec["output_root"]) / spec["action"] / spec["experiment_id"]


def _hydra_config_root(spec: dict[str, Any]) -> Path:
    default = _project_root() / "fullbody" / "config_specific_task" / "posttrain"
    return _as_path(spec.get("hydra_config_root", default))


def _generated_config_name(spec: dict[str, Any], arm_id: str) -> str:
    return f"config_specific_task/posttrain/{spec['action']}/{spec['experiment_id']}/{arm_id}"


def _posttrain_arms(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [arm for arm in spec["arms"] if arm.get("type", "posttrain") != "baseline"]


def _arm_by_id(spec: dict[str, Any], arm_id: str) -> dict[str, Any]:
    for arm in spec["arms"]:
        if arm["id"] == arm_id:
            return arm
    raise ValueError(f"unknown arm: {arm_id}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(base)
    if not override:
        return result
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _env_overrides(spec: dict[str, Any]) -> dict[str, str]:
    env = spec.get("env", {})
    mapping = {
        "MUSCLEMIMIC_AMASS_PATH": env.get("amass_path"),
        "AMASS_PATH": env.get("amass_path"),
        "MUSCLEMIMIC_CONVERTED_AMASS_PATH": env.get("converted_amass_path"),
        "CONVERTED_AMASS_PATH": env.get("converted_amass_path"),
        "MUSCLEMIMIC_SMPL_MODEL_PATH": env.get("smpl_model_path"),
        "SMPL_MODEL_PATH": env.get("smpl_model_path"),
        "MPLCONFIGDIR": env.get("matplotlib_cache", "/tmp/matplotlib"),
        "XDG_CACHE_HOME": env.get("xdg_cache_home", "/tmp"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": str(env.get("xla_python_client_preallocate", "false")).lower(),
    }
    return {key: str(value) for key, value in mapping.items() if value is not None}


def _dataset_conf(motions: list[str], training: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_group": None,
        "rel_dataset_path": motions,
        "retargeting_method": "gmr",
        "clear_cache": False,
        "gmr_config": {
            "src_human": "smplh",
            "target_fps": int(training.get("target_fps", 100)),
            "solver": training.get("solver", "daqp"),
            "damping": float(training.get("damping", 1.0)),
            "offset_to_ground": False,
            "use_velocity_limit": bool(training.get("use_velocity_limit", True)),
            "use_fitted_shape": True,
            "shape_fitting_iterations": int(training.get("shape_fitting_iterations", 500)),
            "ik_config_path": training.get(
                "ik_config_path",
                "loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody_smooth_train.json",
            ),
        },
    }


def build_hydra_config(spec: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    training = _deep_merge(spec.get("training", {}), arm.get("training", {}))
    env_spec = _deep_merge(DEFAULT_ENV, spec.get("env_params", {}))
    reward_params = _deep_merge(DEFAULT_BASE_REWARD, spec.get("reward", {}))
    reward_params = _deep_merge(reward_params, arm.get("reward", {}))
    terminal_params = _deep_merge(DEFAULT_TERMINAL, spec.get("terminal", {}))
    terminal_params = _deep_merge(terminal_params, arm.get("terminal", {}))
    arm_checkpoint_root = str(_as_path(spec.get("checkpoint_root", _output_dir(spec) / "checkpoints")) / arm["id"])

    return {
        "defaults": [f"/{training.get('base_config', 'conf_fullbody_gmr')}", "_self_"],
        "hydra": {"job": {"env_set": _env_overrides(spec)}},
        "wandb": {
            "project": training.get("wandb_project", "musclemimic"),
            "mode": training.get("wandb_mode", "disabled"),
            "tags": ["fullbody", "gmr", "badminton", spec["action"], "posttrain", arm["id"]],
        },
        "experiment": {
            "env_params": {
                "env_name": env_spec["env_name"],
                "num_envs": int(training.get("num_envs", 4096)),
                "disable_fingers": bool(env_spec["disable_fingers"]),
                "goal_params": {"include_current_root_error": False},
                "reward_params": reward_params,
                "terminal_state_type": env_spec["terminal_state_type"],
                "terminal_state_params": terminal_params,
            },
            "checkpoint_root": arm_checkpoint_root,
            "resume_from": str(_as_path(spec["resume_from"])),
            "reset_lr_schedule_on_resume": bool(training.get("reset_lr_schedule_on_resume", True)),
            "reset_logging_timestep": bool(training.get("reset_logging_timestep", False)),
            "lr": training.get("lr", 5e-5),
            "total_timesteps": int(training.get("total_timesteps", 200_000_000)),
            "ppo_config": {
                "num_steps": int(training.get("num_steps", 64)),
                "update_epochs": int(training.get("update_epochs", 2)),
                "num_minibatches": int(training.get("num_minibatches", 128)),
                "init_std": float(training.get("init_std", 1.0)),
                "ent_coef": float(training.get("ent_coef", 0.0)),
            },
            "adaptive_sampling": {"enabled": False},
            "adaptive_termination": {"enabled": False},
            "reward_curriculum": {"enabled": False},
            "asi": {"enabled": False},
            "task_factory": {
                "params": {"amass_dataset_conf": _dataset_conf(spec["reference"]["train"], training)}
            },
            "validation": {
                "active": True,
                "deterministic": bool(training.get("validation_deterministic", True)),
                "num_steps": int(training.get("validation_num_steps", 500)),
                "num_envs": int(training.get("validation_num_envs", 8)),
                "num": int(training.get("validation_num", 8)),
                "video_length": int(training.get("validation_video_length", 500)),
                "video_frequency": int(training.get("validation_video_frequency", 1)),
                "terminal_state_type": env_spec["terminal_state_type"],
                "terminal_state_params": terminal_params,
                "amass_dataset_conf": _dataset_conf(spec["reference"]["validation"], training),
            },
        },
    }


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# @package _global_\n\n" + yaml.safe_dump(data, sort_keys=False))


def _quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _write_command(path: Path, command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_quote_command(command) + "\n")


def prepare_experiment(spec: dict[str, Any]) -> PrepareResult:
    """Materialize configs, command files, and a report for an experiment."""
    output_dir = _output_dir(spec)
    configs_dir = output_dir / "configs"
    hydra_root = _hydra_config_root(spec) / spec["action"] / spec["experiment_id"]
    generated: dict[str, GeneratedConfig] = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "commands").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "spec_snapshot.yaml").write_text(yaml.safe_dump(_strip_private_keys(spec), sort_keys=False))

    for arm in _posttrain_arms(spec):
        config = build_hydra_config(spec, arm)
        output_copy = configs_dir / f"{arm['id']}.yaml"
        hydra_config = hydra_root / f"{arm['id']}.yaml"
        _write_yaml(output_copy, config)
        _write_yaml(hydra_config, config)
        generated_config = GeneratedConfig(
            arm_id=arm["id"],
            config_name=_generated_config_name(spec, arm["id"]),
            output_copy=output_copy,
            hydra_config=hydra_config,
        )
        generated[arm["id"]] = generated_config
        _write_command(output_dir / "commands" / f"train_{arm['id']}.sh", build_train_command(spec, arm["id"], generated_config))

    for arm in spec["arms"]:
        _write_command(output_dir / "commands" / f"eval_{arm['id']}.sh", build_eval_command(spec, arm["id"], render=False))
        _write_command(output_dir / "commands" / f"render_{arm['id']}.sh", build_eval_command(spec, arm["id"], render=True))

    report_path = output_dir / "reports" / "posttrain_plan.md"
    report_path.write_text(_build_report(spec, generated))
    return PrepareResult(output_dir=output_dir, report_path=report_path, generated_configs=generated)


def _strip_private_keys(spec: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in spec.items() if not key.startswith("_")}


def _build_report(spec: dict[str, Any], generated: dict[str, GeneratedConfig]) -> str:
    lines = [
        f"# {spec['action']} {spec['experiment_id']} PostTrain Plan",
        "",
        f"- Resume checkpoint: `{spec['resume_from']}`",
        f"- Train motions: {', '.join(spec['reference']['train'])}",
        f"- Validation motions: {', '.join(spec['reference']['validation'])}",
        f"- Stress motions: {', '.join(spec['reference'].get('stress_test', []))}",
        "",
        "## Arms",
        "",
    ]
    for arm in spec["arms"]:
        lines.append(f"- `{arm['id']}`: {arm.get('description', arm.get('type', 'posttrain'))}")
    lines.extend(["", "## Generated Configs", ""])
    for arm_id, generated_config in generated.items():
        lines.append(f"- `{arm_id}`: `{generated_config.output_copy}`")
    lines.append("")
    return "\n".join(lines)


def build_train_command(spec: dict[str, Any], arm_id: str, generated_config: GeneratedConfig | None = None) -> list[str]:
    if generated_config is None:
        generated_config = GeneratedConfig(
            arm_id=arm_id,
            config_name=_generated_config_name(spec, arm_id),
            output_copy=_output_dir(spec) / "configs" / f"{arm_id}.yaml",
            hydra_config=_hydra_config_root(spec) / spec["action"] / spec["experiment_id"] / f"{arm_id}.yaml",
        )
    training = spec.get("training", {})
    return [
        "uv",
        "run",
        "fullbody/experiment.py",
        f"--config-name={generated_config.config_name}",
        f"wandb.mode={training.get('wandb_mode', 'disabled')}",
    ]


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("checkpoint_")
    try:
        return (int(suffix), path.name)
    except ValueError:
        return (-1, path.name)


def _latest_checkpoint(path: Path) -> Path | None:
    if not path.exists():
        return None
    candidates = [item for item in path.iterdir() if item.is_dir() and item.name.startswith("checkpoint_")]
    if not candidates:
        return None
    return max(candidates, key=_checkpoint_sort_key)


def _checkpoint_for_arm(spec: dict[str, Any], arm: dict[str, Any]) -> str:
    if arm.get("checkpoint"):
        return str(_as_path(arm["checkpoint"]))
    if arm.get("type") == "baseline":
        return str(_as_path(spec["resume_from"]))
    checkpoint_dir = _as_path(spec.get("checkpoint_root", _output_dir(spec) / "checkpoints")) / arm["id"]
    latest = _latest_checkpoint(checkpoint_dir)
    if latest is not None:
        return str(latest)
    return str(checkpoint_dir / "checkpoint_latest")


def build_eval_command(spec: dict[str, Any], arm_id: str, *, render: bool) -> list[str]:
    arm = _arm_by_id(spec, arm_id)
    eval_spec = spec.get("eval", {})
    motion = _normalize_motion_path(eval_spec.get("render_motion") or spec["reference"]["validation"][0])
    command = [
        "uv",
        "run",
        "fullbody/eval.py",
        "--path",
        _checkpoint_for_arm(spec, arm),
        "--motion_path",
        motion,
        "--use_mujoco",
        "--eval_seed",
        str(eval_spec.get("eval_seed", 0)),
        "--n_steps",
        str(eval_spec.get("n_steps", 1000)),
    ]
    if eval_spec.get("stochastic", True):
        command.append("--stochastic")
    if eval_spec.get("no_termination", False):
        command.append("--no_termination")
    if not render:
        command.extend(["--metrics", "--metrics_only", "--metrics_steps", str(eval_spec.get("metrics_steps", 1))])
    else:
        command.extend(["--record", "--record_dir", str(_output_dir(spec) / "videos" / arm_id)])
    return command


def _run_or_print(command: list[str], *, execute: bool, env: dict[str, str] | None = None) -> int:
    print(_quote_command(command))
    if not execute:
        return 0
    completed = subprocess.run(command, cwd=_project_root(), env=env, check=False)
    return int(completed.returncode)


def _env_for_execution(spec: dict[str, Any]) -> dict[str, str]:
    import os

    env = os.environ.copy()
    env.update(_env_overrides(spec))
    if spec.get("eval", {}).get("mujoco_gl"):
        env["MUJOCO_GL"] = str(spec["eval"]["mujoco_gl"])
    return env


def run_stage(spec: dict[str, Any], *, stage: str, arm: str | None, execute: bool) -> int:
    result = prepare_experiment(spec)
    arms = [arm] if arm else [item["id"] for item in spec["arms"]]
    env = _env_for_execution(spec)
    exit_code = 0

    if stage == "prepare":
        print(f"Wrote PostTrain experiment to {result.output_dir}")
        print(f"Report: {result.report_path}")
        return 0

    if stage in {"train", "all"}:
        for arm_id in arms:
            item = _arm_by_id(spec, arm_id)
            if item.get("type") == "baseline":
                continue
            exit_code |= _run_or_print(
                build_train_command(spec, arm_id, result.generated_configs[arm_id]),
                execute=execute,
                env=env,
            )

    if stage in {"eval", "all"}:
        for arm_id in arms:
            exit_code |= _run_or_print(build_eval_command(spec, arm_id, render=False), execute=execute, env=env)

    if stage in {"render", "all"}:
        for arm_id in arms:
            exit_code |= _run_or_print(build_eval_command(spec, arm_id, render=True), execute=execute, env=env)

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="Path to a PostTrain experiment YAML spec")
    parser.add_argument("--stage", choices=("prepare", "train", "eval", "render", "all"), default="prepare")
    parser.add_argument("--arm", default=None, help="Optional arm id to run for train/eval/render")
    parser.add_argument("--execute", action="store_true", help="Actually execute generated commands; default is dry-run")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    return run_stage(spec, stage=args.stage, arm=args.arm, execute=args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
