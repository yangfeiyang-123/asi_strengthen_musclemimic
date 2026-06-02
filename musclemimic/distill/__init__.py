"""Policy distillation utilities for trajectory-conditioned teachers."""

from musclemimic.distill.dataset import DistillDataset, load_metadata, write_distill_shard
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
    "build_student_obs_indices",
    "filter_student_obs",
    "load_metadata",
    "write_distill_shard",
]
