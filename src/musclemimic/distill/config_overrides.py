"""Configuration overrides shared by distillation collection entrypoints."""

from __future__ import annotations

from typing import Any

from musclemimic.runner.eval_utils import apply_trajectory_selection


def apply_collection_overrides(
    config: Any,
    *,
    motion_path: list[str] | None = None,
    motion_group: str | None = None,
    traj_index: int | None = None,
    traj_start_step: int | None = None,
) -> None:
    """Apply motion and fixed-start overrides using eval-compatible semantics."""
    if traj_start_step is not None and traj_index is None:
        raise ValueError("--traj_start_step requires --traj_index")

    dataset_conf = config.experiment.task_factory.params.amass_dataset_conf
    if motion_path:
        dataset_conf.rel_dataset_path = list(motion_path)
        dataset_conf.dataset_group = None
    elif motion_group is not None:
        dataset_conf.dataset_group = str(motion_group)

    apply_trajectory_selection(config, traj_index, traj_start_step)
