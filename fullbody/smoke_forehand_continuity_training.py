#!/usr/bin/env python3
"""Run the canonical three-update continuity reward GPU smoke."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source_repository_environment() -> None:
    """Import exported values from configs/env.sh without printing secrets."""

    command = "source configs/env.sh >/dev/null && env -0"
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    for entry in completed.stdout.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        raw_name, raw_value = entry.split(b"=", 1)
        name = os.fsdecode(raw_name)
        value = os.fsdecode(raw_value)
        os.environ.setdefault(name, value)


_source_repository_environment()

from musclemimic.utils.runtime_env import reexec_with_configured_cuda_env  # noqa: E402

reexec_with_configured_cuda_env()

from hydra import compose, initialize_config_dir  # noqa: E402

from musclemimic.physiology.release import load_continuity_training_release  # noqa: E402
from musclemimic.runner.continuity_smoke import (  # noqa: E402
    load_continuity_training_smoke,
    repository_git_commit,
    resolved_training_config_sha256,
    validate_continuity_training_smoke,
)


def _normalize_config_name(value: str) -> str:
    name = value.strip()
    if name.startswith("fullbody/"):
        name = name.removeprefix("fullbody/")
    if name.endswith(".yaml"):
        name = name[:-5]
    if not name:
        raise ValueError("--config-name cannot be empty")
    return name


def _single_visible_gpu() -> str:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not value or "," in value:
        raise ValueError("CUDA_VISIBLE_DEVICES must select exactly one physical GPU")
    return value


def _forbid_smoke_override_collisions(overrides: list[str]) -> None:
    protected = (
        "experiment.total_timesteps",
        "experiment.num_envs",
        "experiment.env_params.num_envs",
        "experiment.ppo_config.num_minibatches",
        "experiment.training_smoke",
        "experiment.training_root",
        "experiment.checkpoint_root",
        "experiment.run_id",
        "experiment.auto_resume",
        "experiment.resume_from",
        "experiment.validation.active",
        "experiment.promotion.auto_stop",
        "hydra.run.dir",
    )
    for override in overrides:
        key = override.lstrip("+").split("=", 1)[0]
        if any(key == item or key.startswith(f"{item}.") for item in protected):
            raise ValueError(f"smoke driver owns Hydra override {key!r}")


def _run_and_tee(command: list[str], *, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
        except KeyboardInterrupt:
            process.send_signal(2)
            raise
        return process.wait()


def run_smoke(args: argparse.Namespace) -> dict:
    if args.num_updates != 3 or args.num_envs != 8:
        raise ValueError("formal continuity smoke evidence requires --num-updates 3 --num-envs 8")
    _forbid_smoke_override_collisions(args.overrides)
    physical_gpu = _single_visible_gpu()
    output = args.output_json.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite smoke artifact: {output}")

    release = load_continuity_training_release(args.continuity_release)
    os.environ["MUSCLEMIMIC_CONTINUITY_RELEASE"] = str(Path(args.continuity_release).expanduser().resolve())
    os.environ["MUSCLEMIMIC_CONTINUITY_RELEASE_FINGERPRINT"] = release.release_fingerprint
    os.environ["MUSCLEMIMIC_CONTINUITY_SMOKE_ARTIFACT"] = str(output)
    os.environ.pop("MUSCLEMIMIC_CONTINUITY_SMOKE_ARTIFACT_FINGERPRINT", None)

    config_name = _normalize_config_name(args.config_name)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "fullbody")):
        formal_config = compose(config_name=config_name, overrides=args.overrides)
    formal_hash = resolved_training_config_sha256(formal_config)
    formal_experiment = formal_config.experiment
    ablation = formal_experiment.get("continuity_ablation", {})
    condition = str(ablation.get("condition", ""))
    if condition not in {"A1", "B1", "C1", "G1"}:
        raise ValueError("GPU continuity reward smoke requires an A1/B1/C1/G1 config")
    if bool(ablation.get("continuity_reward_enabled", False)) is not True:
        raise ValueError("GPU continuity reward smoke config does not enable the continuity reward")
    if str(formal_experiment.env_params.reward_params.intra_muscle_consistency.mode) != "reward":
        raise ValueError("GPU continuity reward smoke requires reward mode")

    commit = repository_git_commit(ROOT, require_clean=True)
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    work_root = output.parent / f".{output.stem}.work-{uuid.uuid4().hex[:8]}"
    run_id = f"smoke_{condition.lower()}_{uuid.uuid4().hex[:10]}"
    num_steps = int(formal_experiment.ppo_config.num_steps)
    total_timesteps = args.num_updates * args.num_envs * num_steps
    smoke_overrides = [
        f"experiment.env_params.num_envs={args.num_envs}",
        f"experiment.num_envs={args.num_envs}",
        f"experiment.total_timesteps={total_timesteps}",
        f"experiment.ppo_config.num_minibatches={args.num_envs}",
        "experiment.validation.active=false",
        "experiment.promotion.auto_stop=false",
        "experiment.checkpoints_on_validation=false",
        "experiment.save_checkpoints=true",
        "experiment.async_checkpointing=false",
        "experiment.max_checkpoints_to_keep=4",
        "experiment.auto_resume=false",
        "experiment.resume_from=null",
        f"experiment.run_id={run_id}",
        f"experiment.training_root={work_root / 'training'}",
        "experiment.training_smoke.enabled=true",
        f"experiment.training_smoke.output_json={output}",
        f"experiment.training_smoke.formal_config_name={config_name}",
        f"experiment.training_smoke.formal_resolved_config_sha256={formal_hash}",
        f"experiment.training_smoke.formal_run_id={formal_experiment.run_id}",
        f"experiment.training_smoke.started_at_utc={started_at}",
        f"hydra.run.dir={work_root / 'hydra'}",
        "wandb.mode=disabled",
    ]

    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = physical_gpu
    child_env["MM_CUDA_VISIBLE_DEVICES"] = physical_gpu
    child_env.setdefault("MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB", "4")
    child_env.setdefault("MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB", "4")
    cache_key = child_env.get("MUSCLEMIMIC_JAX_CACHE_KEY") or f"continuity_smoke_{condition.lower()}"
    child_env["MUSCLEMIMIC_JAX_CACHE_KEY"] = cache_key
    child_env.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        f"/data3/yangfeiyang/WorkSpace/ENV/jax-cache/{cache_key}",
    )
    log_path = args.log_path.expanduser().resolve() if args.log_path else output.with_suffix(".log")
    command = [
        str(ROOT / "scripts/run_with_cuda_compat.sh"),
        "uv",
        "run",
        "--locked",
        "fullbody/experiment.py",
        f"--config-name={config_name}",
        *args.overrides,
        *smoke_overrides,
    ]
    print(f"[smoke] commit={commit}")
    print(f"[smoke] physical_gpu={physical_gpu}")
    print(f"[smoke] formal_config_sha256={formal_hash}")
    print(f"[smoke] work_root={work_root}")
    print(f"[smoke] log={log_path}")
    exit_code = _run_and_tee(command, env=child_env, log_path=log_path)
    if exit_code != 0:
        raise RuntimeError(f"continuity training smoke process exited with status {exit_code}; see {log_path}")

    artifact = load_continuity_training_smoke(output)
    action_mode = str(ablation.get("action_mode", ""))
    action_config = formal_experiment.get("action_representation", {})
    expected_basis = (
        None if action_mode == "full_354" else str(action_config.get("expected_basis_fingerprint", "") or "")
    )
    return validate_continuity_training_smoke(
        artifact,
        expected_commit_sha=commit,
        expected_resolved_config_sha256=formal_hash,
        expected_release_fingerprint=release.release_fingerprint,
        expected_basis_fingerprint=expected_basis,
        expected_action_mode=action_mode,
        expected_condition=condition,
        max_age_hours=24.0,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--continuity-release", required=True, type=Path)
    parser.add_argument("--num-updates", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional formal Hydra overrides; smoke-owned execution overrides are forbidden.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    artifact = run_smoke(args)
    print(f"[smoke] passed artifact={args.output_json.expanduser().resolve()}")
    print(f"[smoke] artifact_fingerprint={artifact['artifact_fingerprint']}")


if __name__ == "__main__":
    main()
