"""Prepare or launch the standalone Stage-3 LAB incoming-hit trainer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from musclemimic.latent_muscle.checkpoint import load_latent_checkpoint


def build_lab_ppo_manifest(
    *,
    latent_checkpoint_dir: str,
    highlevel_config: str,
    output_dir: str,
    lambda_lab: float,
    correction_actuator_names: list[str] | None = None,
    dry_run: bool = True,
    num_envs: int = 512,
    rollout_steps: int = 64,
    total_env_steps: int | None = None,
    impl: str = "warp",
    resume_from: str | None = None,
    allow_unpromoted_latent: bool = False,
) -> Path:
    checkpoint = load_latent_checkpoint(latent_checkpoint_dir)
    promotion = dict(checkpoint.get("eval_metrics", {}).get("promotion", {}) or {})
    if promotion.get("passed") is not True and not allow_unpromoted_latent:
        raise ValueError("latent checkpoint has not passed its production promotion gate")
    action_mask = checkpoint["action_mask"]
    if correction_actuator_names is not None and list(correction_actuator_names) != action_mask["correction_actuator_names"]:
        raise ValueError(
            "correction actuator partition mismatch: "
            f"checkpoint={action_mask['correction_actuator_names']} runtime={correction_actuator_names}"
        )
    if (
        int(action_mask.get("decoder_action_dim", -1)),
        int(action_mask.get("correction_action_dim", -1)),
        int(action_mask.get("neutral_action_dim", -1)),
    ) != (354, 31, 31):
        raise ValueError("Stage-3 LAB requires a strict 354 body + 31 right + 31 left mask")

    spec_path = Path(highlevel_config)
    if not spec_path.is_file():
        raise FileNotFoundError(f"Stage-3 runner spec not found: {spec_path}")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    lab_config = dict(spec.get("stage3_lab", {}) or {})
    if not bool(lab_config.get("enabled", False)):
        raise ValueError("Stage-3 runner spec must enable stage3_lab")
    configured_lambda = float(
        dict(lab_config.get("curriculum", {}) or {}).get("lambda_start", 0.25)
    )
    if abs(configured_lambda - float(lambda_lab)) > 1e-12:
        raise ValueError(
            f"lambda_lab={lambda_lab} differs from spec curriculum start={configured_lambda}"
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "musclemimic.badminton.scripts.run_incoming_shuttle_hit",
        "--spec",
        str(spec_path),
        "--stage",
        "train-gpu",
        "--latent-checkpoint",
        str(latent_checkpoint_dir),
        "--out-dir",
        str(out),
        "--num-envs",
        str(int(num_envs)),
        "--rollout-steps",
        str(int(rollout_steps)),
        "--impl",
        str(impl),
    ]
    if total_env_steps is not None:
        command.extend(["--total-env-steps", str(int(total_env_steps))])
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from)])
    if allow_unpromoted_latent:
        command.append("--allow-unpromoted-latent")
    payload = {
        "schema_version": "stage3_lab_standalone_v2",
        "git_commit": _git_commit(),
        "latent_checkpoint_dir": latent_checkpoint_dir,
        "highlevel_config": highlevel_config,
        "lambda_lab": float(lambda_lab),
        "dry_run": bool(dry_run),
        "action_mask": action_mask,
        "command": command,
        "promotion": promotion,
    }
    manifest_path = out / "lab_highlevel_ppo_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except Exception:
        return "unknown"
    return result.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent_checkpoint_dir", required=True)
    parser.add_argument("--highlevel_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lambda_lab", type=float, default=0.25)
    parser.add_argument("--correction_actuator_names", nargs="*", default=None)
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--rollout_steps", type=int, default=64)
    parser.add_argument("--total_env_steps", type=int, default=None)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--allow_unpromoted_latent", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_lab_ppo_manifest(
        latent_checkpoint_dir=args.latent_checkpoint_dir,
        highlevel_config=args.highlevel_config,
        output_dir=args.output_dir,
        lambda_lab=float(args.lambda_lab),
        correction_actuator_names=args.correction_actuator_names,
        dry_run=bool(args.dry_run),
        num_envs=int(args.num_envs),
        rollout_steps=int(args.rollout_steps),
        total_env_steps=args.total_env_steps,
        impl=str(args.impl),
        resume_from=args.resume_from,
        allow_unpromoted_latent=bool(args.allow_unpromoted_latent),
    )
    print(f"lab_highlevel_ppo_manifest: {manifest}")
    if not args.dry_run:
        subprocess.run(json.loads(manifest.read_text(encoding="utf-8"))["command"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
