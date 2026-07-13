from __future__ import annotations

from typing import Any

from musclemimic.runner.validation_video_recorder import ValidationVideoRecorder


class ExperimentHooks:
    """Hooks to customize logging and video behavior across experiments."""

    def build_video_recorder(self, result_dir: str, config) -> ValidationVideoRecorder | None:
        val = getattr(config, "experiment", {}).get("validation", {}) if hasattr(config, "experiment") else {}
        active = bool(val.get("active", False))
        if not active:
            return None
        frequency = int(val.get("video_frequency", 10))
        length = int(val.get("video_length", 250))
        deterministic = bool(val.get("deterministic", True))
        cycle_trajectories = bool(val.get("cycle_video_trajectories", False))
        max_recordings = val.get("video_max_recordings", None)
        review_kind = val.get("visual_review_kind", None)
        return ValidationVideoRecorder(
            video_dir=result_dir,
            frequency=frequency,
            length=length,
            deterministic=deterministic,
            cycle_trajectories=cycle_trajectories,
            max_recordings=max_recordings,
            review_kind=review_kind,
        )

    def enrich_log(self, log_dict: dict[str, Any], metrics_dict: dict[str, Any], env) -> None:
        """Opportunity to add environment-specific fields to logs."""
        # Default: no-op
        return None

    def on_validation_video(self, use_wandb: bool, wandb, video_path: str | None, timestep: int) -> None:
        if not use_wandb or not video_path:
            return
        try:
            wandb.log({"Validation/Video": wandb.Video(video_path, format="mp4")}, step=int(timestep))
        except Exception:
            # Keep evaluation robust; do not raise during logging
            print("Warning: failed to log video to wandb.")


class UnifiedHooks(ExperimentHooks):
    """Unified hooks used by both fullbody and bimanual experiments."""

    pass
