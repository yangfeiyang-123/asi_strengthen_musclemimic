"""Host-side promotion progress for Stage-1/Stage-2 PPO early stopping.

The PPO update itself remains compiled.  This module owns only the small amount
of host state that must survive process restarts: validation-boundary history,
the consecutive-pass streak, and the final stop reason.  Numeric acceptance is
delegated to :mod:`musclemimic.badminton.training_gates`, so the online and
offline promotion decisions cannot silently diverge.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from musclemimic.badminton.training_gates import (
    evaluate_promotion,
    latest_validation_record,
    validate_promotion_threshold_config,
)

_SCHEMA_VERSION = "forehand_clear_promotion_progress_v3"
_SUPPORTED_STAGES = frozenset({"stage1", "stage2"})


@dataclass(frozen=True)
class PromotionEarlyStopSettings:
    """Resolved, fail-closed configuration for online promotion decisions."""

    stage: str
    consecutive_required: int
    min_validations_before_stop: int
    progress_path: Path
    checkpoint_dir: Path
    config_hash: str
    baseline_metrics_path: Path | None = None
    baseline_metrics_sha256: str | None = None
    baseline_metrics: Mapping[str, Any] | None = None


def resolve_promotion_early_stop(config: Any) -> PromotionEarlyStopSettings | None:
    """Resolve early-stop settings, returning ``None`` for the ordinary path.

    Enabling host-controlled early stopping with multiple seeds is rejected
    rather than silently falling back to a full-budget run.  A Stage-2 baseline
    is mandatory because its body-degradation gate is otherwise undefined.
    """

    promotion = config.get("promotion", {})
    if not bool(promotion.get("auto_stop", False)):
        return None

    n_seeds = int(config.get("n_seeds", 1))
    if n_seeds != 1:
        raise ValueError(
            "promotion.auto_stop=true is supported only for a single seed; "
            f"got experiment.n_seeds={n_seeds}"
        )
    validation = config.get("validation", {})
    if not bool(validation.get("active", False)):
        raise ValueError("promotion.auto_stop=true requires experiment.validation.active=true")
    if not bool(validation.get("cover_all_trajectories", False)):
        raise ValueError(
            "promotion.auto_stop=true requires "
            "experiment.validation.cover_all_trajectories=true"
        )
    if not bool(validation.get("deterministic", False)):
        raise ValueError(
            "promotion.auto_stop=true requires deterministic held-out evaluation"
        )
    if not bool(validation.get("start_from_beginning", False)):
        raise ValueError(
            "promotion.auto_stop=true requires validation.start_from_beginning=true"
        )
    dataset_cfg = validation.get("amass_dataset_conf", {})
    motion_paths = list(dataset_cfg.get("rel_dataset_path", []) or [])
    validation_num_envs = int(validation.get("num_envs", 0) or 0)
    if motion_paths and validation_num_envs < len(motion_paths):
        raise ValueError(
            "validation.num_envs must be at least the held-out trajectory count "
            "when cover_all_trajectories=true"
        )

    stage = str(promotion.get("stage", "")).lower()
    if stage not in _SUPPORTED_STAGES:
        raise ValueError(
            "promotion.auto_stop=true requires promotion.stage to be stage1 or stage2"
        )
    validate_promotion_threshold_config(stage, promotion)
    required = int(promotion.get("consecutive_validations", 3))
    if required <= 0:
        raise ValueError("promotion.consecutive_validations must be positive")
    min_validations = int(promotion.get("min_validations_before_stop", required))
    if min_validations < required:
        raise ValueError(
            "promotion.min_validations_before_stop must be at least "
            "promotion.consecutive_validations"
        )

    checkpoint_dir = Path(str(config.get("checkpoint_dir", "checkpoints"))).expanduser().resolve()
    progress_value = promotion.get("progress_path", None)
    progress_path = (
        checkpoint_dir / "promotion_progress.json"
        if progress_value in (None, "")
        else Path(str(progress_value)).expanduser().resolve()
    )

    save_checkpoints = bool(config.get("save_checkpoints", False))
    save_on_validation = bool(
        config.get(
            "save_checkpoints_on_validation",
            config.get("checkpoints_on_validation", True),
        )
    )
    if not (save_checkpoints and save_on_validation):
        raise ValueError(
            "promotion.auto_stop=true requires validation-boundary checkpointing "
            "(save_checkpoints=true and checkpoints_on_validation=true)"
        )

    baseline_path: Path | None = None
    baseline_sha256: str | None = None
    baseline: Mapping[str, Any] | None = None
    if stage == "stage2":
        baseline_value = promotion.get("baseline_metrics_path", None)
        if baseline_value in (None, ""):
            raise ValueError(
                "Stage-2 promotion.auto_stop requires promotion.baseline_metrics_path "
                "pointing to accepted Stage-1 validation metrics"
            )
        baseline_path = Path(str(baseline_value)).expanduser().resolve()
        if not baseline_path.is_file():
            raise ValueError(f"Stage-2 baseline metrics do not exist: {baseline_path}")
        baseline_sha256 = _sha256_file(baseline_path)
        baseline = _load_latest_validation(baseline_path)

    # Lazy import avoids a package cycle during lightweight distillation
    # imports: algorithms.ppo.runner -> this module -> runner package -> video
    # recorder -> algorithms.PPOJax.
    from musclemimic.runner.checkpointing import config_hash

    return PromotionEarlyStopSettings(
        stage=stage,
        consecutive_required=required,
        min_validations_before_stop=min_validations,
        progress_path=progress_path,
        checkpoint_dir=checkpoint_dir,
        config_hash=config_hash(config),
        baseline_metrics_path=baseline_path,
        baseline_metrics_sha256=baseline_sha256,
        baseline_metrics=baseline,
    )


def validation_chunk_length(
    current_update: int,
    remaining_updates: int,
    validation_interval: int,
) -> int:
    """Return a positive scan length ending at the next validation boundary."""

    current = int(current_update)
    remaining = int(remaining_updates)
    interval = int(validation_interval)
    if remaining <= 0:
        return 0
    if interval <= 0:
        raise ValueError("validation_interval must be positive")
    distance = interval - (current % interval)
    return min(remaining, distance)


def load_promotion_progress(
    settings: PromotionEarlyStopSettings,
    *,
    checkpoint_update: int,
) -> dict[str, Any]:
    """Load progress and roll it back to the checkpoint being resumed.

    The history filter makes a manually selected older checkpoint safe: a JSON
    record written by a newer checkpoint cannot donate its pass streak to the
    older policy.
    """

    _validate_baseline_unchanged(settings)
    if settings.progress_path.is_file():
        try:
            payload = json.loads(settings.progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"promotion progress is unreadable: {settings.progress_path}"
            ) from exc
        _validate_progress_identity(payload, settings)
        raw_history = payload.get("history", [])
        if not isinstance(raw_history, list):
            raise ValueError("promotion progress history must be a JSON list")
    else:
        raw_history = []

    history = [
        event
        for event in raw_history
        if isinstance(event, Mapping)
        and int(event.get("update_number", -1)) <= int(checkpoint_update)
    ]
    streak = _tail_pass_streak(history)
    stopped = bool(
        streak >= settings.consecutive_required
        and len(history) >= settings.min_validations_before_stop
        and history
        and int(history[-1]["update_number"]) == int(checkpoint_update)
    )
    return _build_progress(
        settings,
        history=history,
        streak=streak,
        stopped_early=stopped,
        hard_cap_reached=False,
    )


def record_validation(
    settings: PromotionEarlyStopSettings,
    progress: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any],
    update_number: int,
    global_timestep: int,
    checkpoint_identity: Mapping[str, Any],
    validation_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one validation, append it, and atomically persist progress."""

    _validate_baseline_unchanged(settings)
    one_report = evaluate_promotion(
        settings.stage,
        metrics,
        consecutive=1,
        baseline_metrics=settings.baseline_metrics,
    )
    event = {
        "update_number": int(update_number),
        "global_timestep": int(global_timestep),
        "passed": bool(one_report.passed),
        "metrics": dict(metrics),
        "gate_report": one_report.to_dict(),
        "checkpoint_identity": dict(checkpoint_identity),
        "validation_provenance": dict(validation_provenance),
    }
    for key in ("update_number", "global_timestep"):
        try:
            identity_value = int(checkpoint_identity[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"promotion checkpoint identity requires integer {key}"
            ) from exc
        if identity_value != int(event[key]):
            raise ValueError(
                f"promotion checkpoint identity {key} differs from validation boundary"
            )
    if checkpoint_identity.get("config_hash") != settings.config_hash:
        raise ValueError("promotion checkpoint identity config hash differs from run")
    if validation_provenance.get("semantics") != "evaluate_all_once_per_heldout_v1":
        raise ValueError("promotion validation must use strict evaluate-all semantics")
    history = list(progress.get("history", []))
    if history and int(history[-1].get("update_number", -1)) >= int(update_number):
        history = [
            item
            for item in history
            if int(item.get("update_number", -1)) < int(update_number)
        ]
    history.append(event)

    streak = _tail_pass_streak(history)
    stopped = bool(
        streak >= settings.consecutive_required
        and len(history) >= settings.min_validations_before_stop
    )
    result = _build_progress(
        settings,
        history=history,
        streak=streak,
        stopped_early=stopped,
        hard_cap_reached=False,
    )
    _atomic_write_json(settings.progress_path, result)
    return result


def finalize_promotion_progress(
    settings: PromotionEarlyStopSettings,
    progress: Mapping[str, Any],
    *,
    hard_cap_reached: bool,
) -> dict[str, Any]:
    """Persist the terminal state after either early stop or the hard cap."""

    _validate_baseline_unchanged(settings)
    history = list(progress.get("history", []))
    streak = _tail_pass_streak(history)
    stopped = bool(progress.get("stopped_early", False)) or (
        streak >= settings.consecutive_required
        and len(history) >= settings.min_validations_before_stop
    )
    result = _build_progress(
        settings,
        history=history,
        streak=streak,
        stopped_early=stopped,
        hard_cap_reached=bool(hard_cap_reached and not stopped),
    )
    _atomic_write_json(settings.progress_path, result)
    return result


def _build_progress(
    settings: PromotionEarlyStopSettings,
    *,
    history: list[Mapping[str, Any]],
    streak: int,
    stopped_early: bool,
    hard_cap_reached: bool,
) -> dict[str, Any]:
    latest = history[-1] if history else None
    return {
        "schema_version": _SCHEMA_VERSION,
        "stage": settings.stage,
        "checkpoint_dir": str(settings.checkpoint_dir),
        "config_hash": settings.config_hash,
        "baseline_metrics_path": (
            None
            if settings.baseline_metrics_path is None
            else str(settings.baseline_metrics_path)
        ),
        "baseline_metrics_sha256": settings.baseline_metrics_sha256,
        "consecutive_required": settings.consecutive_required,
        "min_validations_before_stop": settings.min_validations_before_stop,
        "consecutive_pass_streak": int(streak),
        "validation_count": len(history),
        "last_validation_update": (
            None if latest is None else int(latest["update_number"])
        ),
        "last_global_timestep": (
            None if latest is None else int(latest["global_timestep"])
        ),
        "stopped_early": bool(stopped_early),
        "hard_cap_reached": bool(hard_cap_reached),
        "stop_reason": (
            "consecutive_promotion_pass"
            if stopped_early
            else ("hard_cap" if hard_cap_reached else None)
        ),
        # Compatibility alias consumed by the offline gate and by Stage 2.
        # ``history`` remains authoritative because it also binds each record
        # to its checkpoint update/global timestep and complete gate report.
        "validations": [dict(event["metrics"]) for event in history],
        "history": history,
    }


def _tail_pass_streak(history: list[Mapping[str, Any]]) -> int:
    streak = 0
    for event in reversed(history):
        if not bool(event.get("passed", False)):
            break
        streak += 1
    return streak


def _validate_progress_identity(
    payload: Mapping[str, Any],
    settings: PromotionEarlyStopSettings,
) -> None:
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("promotion progress schema is incompatible")
    if str(payload.get("stage", "")).lower() != settings.stage:
        raise ValueError("promotion progress belongs to a different stage")
    if int(payload.get("consecutive_required", -1)) != settings.consecutive_required:
        raise ValueError("promotion progress uses a different consecutive gate")
    if int(payload.get("min_validations_before_stop", -1)) != settings.min_validations_before_stop:
        raise ValueError("promotion progress uses a different minimum validation count")
    if payload.get("config_hash") != settings.config_hash:
        raise ValueError("promotion progress uses a different training config hash")
    recorded_baseline = payload.get("baseline_metrics_path", None)
    expected_baseline = (
        None
        if settings.baseline_metrics_path is None
        else str(settings.baseline_metrics_path)
    )
    if recorded_baseline != expected_baseline:
        raise ValueError("promotion progress uses a different Stage-1 baseline")
    recorded_baseline_sha256 = payload.get("baseline_metrics_sha256", None)
    if recorded_baseline_sha256 != settings.baseline_metrics_sha256:
        raise ValueError("promotion progress uses different Stage-1 baseline content")
    _validate_baseline_unchanged(settings)


def _load_latest_validation(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"baseline metrics are unreadable: {path}") from exc
    return latest_validation_record(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_baseline_unchanged(settings: PromotionEarlyStopSettings) -> None:
    if settings.baseline_metrics_path is None:
        return
    try:
        current = _sha256_file(settings.baseline_metrics_path)
    except OSError as exc:
        raise ValueError("Stage-1 baseline metrics disappeared during Stage-2") from exc
    if current != settings.baseline_metrics_sha256:
        raise ValueError("Stage-1 baseline metrics changed after Stage-2 startup")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)
