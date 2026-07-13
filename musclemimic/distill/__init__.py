"""Policy distillation utilities for trajectory-conditioned teachers.

``DistillDataset``, ``load_metadata``, ``write_distill_shard``, and
``write_split_shard`` are always importable (pure numpy).  Loss helpers,
observation filters, collectors, and trainers require JAX and are imported
lazily.
"""

from musclemimic.distill.dataset import DistillDataset, load_metadata, write_distill_shard, write_split_shard
from musclemimic.distill.motion_identity import MotionIdentityMap, stable_motion_uid

__all__ = [
    "DistillDataset",
    "MotionIdentityMap",
    "StudentObservationFilterWrapper",
    "StudentObsSpec",
    "bc_loss",
    "build_student_obs_indices",
    "collect_dagger_dataset",
    "collect_teacher_dataset",
    "distribution_log_std",
    "distribution_mean",
    "extract_reference_features",
    "filter_student_obs",
    "gaussian_diag_kl",
    "load_metadata",
    "reference_feature_indices",
    "stable_motion_uid",
    "train_bc",
    "write_distill_shard",
    "write_split_shard",
]


def __getattr__(name):
    """Lazy exports for JAX/env-dependent helpers."""
    _losses = {"bc_loss", "distribution_log_std", "distribution_mean", "gaussian_diag_kl"}
    if name in _losses:
        from musclemimic.distill import losses

        return getattr(losses, name)

    _obs_filter = {
        "StudentObservationFilterWrapper",
        "StudentObsSpec",
        "build_student_obs_indices",
        "extract_reference_features",
        "filter_student_obs",
        "reference_feature_indices",
    }
    if name in _obs_filter:
        from musclemimic.distill import obs_filter

        return getattr(obs_filter, name)

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
