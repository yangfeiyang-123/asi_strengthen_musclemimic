"""Shared experiment runner utilities with cycle-safe lazy exports."""

__all__ = ["ValidationVideoRecorder"]


def __getattr__(name: str):
    # PPO runner imports checkpoint helpers while algorithms itself is still
    # being initialized.  Importing the video recorder eagerly here loops back
    # into ``musclemimic.algorithms.PPOJax`` and breaks every non-video caller.
    if name == "ValidationVideoRecorder":
        from .validation_video_recorder import ValidationVideoRecorder

        return ValidationVideoRecorder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
