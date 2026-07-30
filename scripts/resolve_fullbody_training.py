#!/usr/bin/env python3
"""Resolve and summarize a full-body training launch without constructing JAX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from musclemimic.runner.continuity_smoke import (
    load_continuity_training_smoke,
    repository_git_commit,
    resolved_training_config_sha256,
    validate_continuity_training_smoke,
)

ROOT = Path(__file__).resolve().parents[1]


def _native(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _config_name(value: str) -> str:
    result = value.strip().removeprefix("fullbody/")
    return result[:-5] if result.endswith(".yaml") else result


def _resolved_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _validate_formal_continuity_smoke(
    config: Any,
    *,
    resolved_config_sha256: str,
    action_mode: str,
    condition: str | None,
) -> dict[str, Any]:
    """Validate immutable reward smoke evidence before the training process starts."""

    from musclemimic.physiology.release import load_continuity_training_release

    experiment = config.experiment
    continuity = experiment.env_params.reward_params.intra_muscle_consistency
    release_path = str(continuity.get("release_path", "") or "").strip()
    expected_release = str(continuity.get("expected_release_fingerprint", "") or "").strip()
    if not release_path or not expected_release:
        raise ValueError("formal continuity reward launch must pin one release path and fingerprint")
    release = load_continuity_training_release(_resolved_path(release_path))
    if release.release_fingerprint != expected_release:
        raise ValueError("configured continuity release fingerprint differs from the loaded release")

    gate = experiment.get("continuity_smoke_gate", {})
    if not bool(gate.get("required", False)):
        raise ValueError("formal continuity reward launch must require the GPU smoke gate")
    artifact_path = str(gate.get("artifact_path", "") or "").strip()
    if not artifact_path:
        raise ValueError("formal continuity reward launch has no smoke artifact path")

    action = experiment.get("action_representation", {})
    expected_basis: str | None = None
    if action_mode != "full_354":
        expected_basis = str(action.get("expected_basis_fingerprint", "") or "").strip()
        if not expected_basis:
            raise ValueError("formal synergy launch must pin its basis fingerprint")
    commit = repository_git_commit(
        ROOT,
        require_clean=bool(gate.get("require_clean_git", True)),
    )
    artifact = load_continuity_training_smoke(_resolved_path(artifact_path))
    return validate_continuity_training_smoke(
        artifact,
        expected_commit_sha=commit,
        expected_resolved_config_sha256=resolved_config_sha256,
        expected_release_fingerprint=release.release_fingerprint,
        expected_basis_fingerprint=expected_basis,
        expected_action_mode=action_mode,
        expected_condition=condition,
        expected_artifact_fingerprint=(str(gate.get("expected_artifact_fingerprint", "") or "").strip() or None),
        max_age_hours=float(gate.get("max_age_hours", 24.0)),
    )


def build_training_preflight_summary(
    config_name: str,
    overrides: list[str],
    *,
    validate_continuity_smoke: bool = False,
) -> dict[str, Any]:
    name = _config_name(config_name)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "fullbody")):
        config = compose(config_name=name, overrides=overrides)
    experiment = config.experiment
    ablation = experiment.get("continuity_ablation", {})
    action = experiment.get("action_representation", {})
    action_enabled = bool(action.get("enabled", False))
    action_mode = str(action.get("mode", "full_354")) if action_enabled else "full_354"
    continuity = experiment.get("env_params", {}).get("reward_params", {}).get("intra_muscle_consistency", {})
    continuity_mode = str(continuity.get("mode", "off"))
    release_path = str(continuity.get("release_path", "") or "") or None
    auto_resume = bool(experiment.get("auto_resume", True))
    resume_from = experiment.get("resume_from", None)
    fresh_required = bool(ablation.get("fresh_optimizer_required", False))
    fresh_optimizer = not auto_resume and resume_from is None and (not ablation or fresh_required)
    disable_fingers = bool(experiment.env_params.get("disable_fingers", False))
    ordered_channels = 354 if disable_fingers else None
    if ablation and not fresh_optimizer:
        raise ValueError("continuity ablation preflight requires a fresh optimizer")
    if continuity_mode == "reward" and not release_path:
        raise ValueError("continuity reward preflight requires one immutable release")

    reward = _native(experiment.env_params.reward_params)
    terminal = _native(experiment.env_params.get("terminal_state_params", {}))
    promotion = _native(experiment.get("promotion", {}))
    summary = {
        "schema_version": "fullbody_training_dry_run_preflight_v1",
        "config_name": name,
        "resolved_config_sha256": resolved_training_config_sha256(config),
        "run_id": str(experiment.get("run_id", "") or ""),
        "total_timesteps": int(experiment.total_timesteps),
        "seeds": list(experiment.get("seeds", []) or []),
        "condition": str(ablation.get("condition", "")) or None,
        "action_mode": action_mode,
        "basis_family": str(ablation.get("basis_family", "")) or None,
        "basis_fingerprint": (
            str(action.get("expected_basis_fingerprint", "") or "") or None if action_enabled else None
        ),
        "continuity_mode": continuity_mode,
        "continuity_release": release_path,
        "continuity_release_fingerprint": (str(continuity.get("expected_release_fingerprint", "") or "") or None),
        "disable_fingers": disable_fingers,
        "ordered_muscle_channels": ordered_channels,
        "auto_resume": auto_resume,
        "resume_from": resume_from,
        "optimizer_state": "fresh" if fresh_optimizer else "resume_or_config_dependent",
        "reward_weights": {
            key: value
            for key, value in reward.items()
            if key.endswith("_w_sum") or key.endswith("_coeff") or key == "intra_muscle_consistency"
        },
        "terminal_state_type": str(experiment.env_params.get("terminal_state_type", "")),
        "terminal_thresholds": terminal,
        "promotion": promotion,
    }
    if not summary["run_id"]:
        raise ValueError("training preflight requires an explicit run_id")
    summary["continuity_smoke_validated"] = False
    summary["continuity_smoke_artifact_fingerprint"] = None
    if validate_continuity_smoke and continuity_mode == "reward":
        artifact = _validate_formal_continuity_smoke(
            config,
            resolved_config_sha256=summary["resolved_config_sha256"],
            action_mode=action_mode,
            condition=summary["condition"],
        )
        summary["continuity_smoke_validated"] = True
        summary["continuity_smoke_artifact_fingerprint"] = artifact["artifact_fingerprint"]
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-continuity-smoke",
        action="store_true",
        help="Require a fresh commit/config/release/basis-bound GPU smoke artifact for reward configs.",
    )
    parser.add_argument("--config-name", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(
        json.dumps(
            build_training_preflight_summary(
                args.config_name,
                args.overrides,
                validate_continuity_smoke=args.validate_continuity_smoke,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
