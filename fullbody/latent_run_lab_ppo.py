"""Prepare a high-level LAB PPO run manifest from a latent checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from musclemimic.latent_muscle.checkpoint import load_latent_checkpoint


def build_lab_ppo_manifest(
    *,
    latent_checkpoint_dir: str,
    highlevel_config: str,
    output_dir: str,
    lambda_lab: float,
    correction_actuator_names: list[str] | None = None,
    dry_run: bool = True,
) -> Path:
    checkpoint = load_latent_checkpoint(latent_checkpoint_dir)
    action_mask = checkpoint["action_mask"]
    if correction_actuator_names is not None and list(correction_actuator_names) != action_mask["correction_actuator_names"]:
        raise ValueError(
            "correction actuator partition mismatch: "
            f"checkpoint={action_mask['correction_actuator_names']} runtime={correction_actuator_names}"
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "fullbody/experiment.py",
        f"--config-name={highlevel_config}",
        f"experiment.latent_lab.checkpoint_dir={latent_checkpoint_dir}",
        f"experiment.latent_lab.lambda_lab={float(lambda_lab)}",
    ]
    payload = {
        "schema_version": "lab_highlevel_ppo_v1",
        "git_commit": _git_commit(),
        "latent_checkpoint_dir": latent_checkpoint_dir,
        "highlevel_config": highlevel_config,
        "lambda_lab": float(lambda_lab),
        "dry_run": bool(dry_run),
        "action_mask": action_mask,
        "command": command,
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
    parser.add_argument("--lambda_lab", type=float, default=1.0)
    parser.add_argument("--correction_actuator_names", nargs="*", default=None)
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=True)
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
    )
    print(f"lab_highlevel_ppo_manifest: {manifest}")
    if not args.dry_run:
        print("Run the recorded command after adding experiment.latent_lab support to the selected Hydra config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
