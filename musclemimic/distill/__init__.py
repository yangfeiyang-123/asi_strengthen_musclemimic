"""Policy distillation utilities for trajectory-conditioned teachers."""

from musclemimic.distill.dataset import DistillDataset, load_metadata, write_distill_shard, write_split_shard
from musclemimic.distill.losses import bc_loss, distribution_log_std, distribution_mean, gaussian_diag_kl
from musclemimic.distill.obs_filter import (
    StudentObservationFilterWrapper,
    StudentObsSpec,
    build_student_obs_indices,
    filter_student_obs,
)

__all__ = [
    "DistillDataset",
    "StudentObservationFilterWrapper",
    "StudentObsSpec",
    "bc_loss",
    "build_student_obs_indices",
    "collect_dagger_dataset",
    "collect_teacher_dataset",
    "distribution_log_std",
    "distribution_mean",
    "filter_student_obs",
    "gaussian_diag_kl",
    "load_metadata",
    "train_bc",
    "write_distill_shard",
    "write_split_shard",
]


def __getattr__(name):
    """Lazy exports for env/PPO-dependent helpers to avoid circular imports."""
    if name == "collect_teacher_dataset":
        from musclemimic.distill.collect_teacher import collect_teacher_dataset

        return collect_teacher_dataset
    if name == "collect_dagger_dataset":
        from musclemimic.distill.dagger import collect_dagger_dataset

        return collect_dagger_dataset
    if name == "train_bc":
        from musclemimic.distill.train_bc import train_bc

        return train_bc
    raise AttributeError(name)
