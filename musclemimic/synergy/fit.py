"""Offline, fail-closed NMF fitting for physical muscle signals.

This module deliberately operates on persisted distillation shards.  It never
constructs an environment or restores a policy checkpoint, so fitting and its
CPU tests cannot interfere with live RL jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict, loads_json_strict
from musclemimic.distill.action_schema import actuator_schema_hash, ordered_schema_hash
from musclemimic.distill.physical import (
    MUSCLE_ACTIVATION_SOURCE,
    PHYSICAL_CAPTURE_SCHEMA_VERSION,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
    UNIT_INTERVAL_ROUNDOFF_POLICY,
    validate_activation_valid_mask,
    validate_muscle_channel_contract,
    validate_physical_signal_semantics,
    validate_unit_muscle_activation,
    validate_unit_muscle_ctrlrange,
)
from musclemimic.synergy.action_interface import save_coefficient_statistics
from musclemimic.synergy.basis_artifact import load_synergy_basis, save_synergy_basis
from musclemimic.synergy.collect import ctrl_to_unit_excitation
from musclemimic.synergy.grouping import global_group, load_grouping_json
from musclemimic.synergy.hybrid_basis import (
    HybridBasisConfig,
    build_hybrid_basis,
    save_hybrid_basis_artifact,
    validate_hybrid_basis_result,
)
from musclemimic.synergy.metrics import basis_condition_number, reconstruction_metrics
from musclemimic.synergy.nmf import NMFResult, fit_best_initialization, fit_nmf, transform_nmf
from musclemimic.synergy.preprocess import (
    DEFAULT_PHASE_WEIGHTS,
    apply_preprocessor,
    apply_sample_weights,
    fit_preprocessor,
    phase_balanced_weights,
)
from musclemimic.synergy.primitive_manifest import load_primitive_source_manifest
from musclemimic.synergy.rank_selection import (
    BasisNotEligibleForEarlyControl,
    candidate_basis_fingerprint,
    candidate_ranks_for_region,
    canonical_candidate_ranks,
    canonical_region_candidate_ranks,
    dynamic_coverage_report_for_rank,
    dynamic_coverage_requirement,
    enforce_total_rank_budget,
    select_smallest_eligible_rank,
    validate_dynamic_coverage_gate,
    validate_dynamic_coverage_rank_inventory,
)
from musclemimic.synergy.schema import (
    ACTIVATION_SIGNAL_KIND,
    EXCITATION_SIGNAL_KIND,
    SignalTransform,
    SynergySignal,
    ctrlrange_schema_hash,
)
from musclemimic.synergy.stability import (
    bootstrap_stability,
    cross_trial_stability,
    initialization_stability,
    split_half_stability,
)

REPORT_SCHEMA_VERSION = "forehand_clear_synergy_fit_report_v1"
_SIGNAL_ALIASES = {
    "excitation": EXCITATION_SIGNAL_KIND,
    "physical_excitation": EXCITATION_SIGNAL_KIND,
    EXCITATION_SIGNAL_KIND: EXCITATION_SIGNAL_KIND,
    "activation": ACTIVATION_SIGNAL_KIND,
    ACTIVATION_SIGNAL_KIND: ACTIVATION_SIGNAL_KIND,
}
_SAMPLED_FIELDS = (
    "teacher_ctrl_physical",
    "muscle_excitation",
    "muscle_activation",
    "phase_id",
    "motion_uid",
    "traj_no",
    "task_id",
    "trial_id",
    "source_kind",
    "success",
    "quality_weight",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SynergyFitConfig:
    ranks: tuple[int, ...] = tuple(range(1, 11))
    region_ranks: Mapping[str, tuple[int, ...]] | None = None
    total_rank_budget: int | None = None
    require_dynamic_coverage: bool = False
    max_mean_dynamic_gap: float = 0.15
    max_key_phase_dynamic_gap: float = 0.25
    expected_environment_fingerprint: str | None = None
    expected_rollout_manifest_fingerprint: str | None = None
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    normalization: str = "channel_max"
    near_zero_threshold: float = 1e-8
    phase_weights: Mapping[int, float] | None = None
    max_iter: int = 1000
    tol: float = 1e-6
    split_half_repeats: int = 5
    bootstrap_repeats: int = 10
    cross_trial_max_trials: int = 12
    min_val_global_vaf: float = 0.90
    min_val_local_vaf_quantile: float = 0.70
    local_vaf_quantile: float = 0.10
    min_initialization_similarity: float = 0.80
    min_split_half_similarity: float = 0.80
    min_bootstrap_similarity: float = 0.80
    min_cross_trial_similarity: float = 0.75
    max_basis_condition_number: float = 1.0e6
    min_effective_rank_fraction: float = 1.0
    hybrid_novelty_residual_ratio: float = 0.15
    hybrid_duplicate_cosine_similarity: float = 0.95
    hybrid_min_heldout_global_vaf_marginal_gain: float = 1e-6
    hybrid_max_total_rank: int = 64
    hybrid_min_heldout_global_vaf: float = 0.90
    hybrid_local_vaf_quantile: float = 0.10
    hybrid_min_heldout_local_vaf_quantile: float = 0.70
    hybrid_max_basis_condition_number: float = 100.0
    hybrid_min_effective_rank_fraction: float = 0.80
    hybrid_effective_rank_relative_tolerance: float = 1e-8

    def validated(self) -> SynergyFitConfig:
        ranks = canonical_candidate_ranks(self.ranks)
        region_ranks = canonical_region_candidate_ranks(self.region_ranks)
        if self.total_rank_budget is not None and (
            isinstance(self.total_rank_budget, bool)
            or not isinstance(self.total_rank_budget, int | np.integer)
            or int(self.total_rank_budget) <= 0
        ):
            raise ValueError("total_rank_budget must be a positive integer or null")
        if type(self.require_dynamic_coverage) is not bool:
            raise ValueError("require_dynamic_coverage must be boolean")
        expected_environment = self.expected_environment_fingerprint
        expected_rollout = self.expected_rollout_manifest_fingerprint
        if (expected_environment is None) != (expected_rollout is None):
            raise ValueError(
                "expected_environment_fingerprint and "
                "expected_rollout_manifest_fingerprint must be pinned together"
            )
        if self.require_dynamic_coverage and expected_environment is None:
            raise ValueError(
                "require_dynamic_coverage=true requires expected environment and "
                "rollout manifest fingerprints"
            )
        if expected_environment is not None:
            expected_environment = _require_sha256(
                expected_environment,
                field="expected_environment_fingerprint",
            )
            expected_rollout = _require_sha256(
                expected_rollout,
                field="expected_rollout_manifest_fingerprint",
            )
        for name in ("max_mean_dynamic_gap", "max_key_phase_dynamic_gap"):
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool):
                raise ValueError(f"{name} must be finite and non-negative")
            value = float(raw_value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        seeds = tuple(int(value) for value in self.seeds)
        if len(seeds) < 2 or any(value < 0 for value in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("at least two distinct non-negative initialization seeds are required")
        if self.normalization not in {"channel_max", "channel_l2", "none"}:
            raise ValueError("normalization must be channel_max, channel_l2, or none")
        if self.near_zero_threshold < 0.0 or self.max_iter <= 0 or self.tol < 0.0:
            raise ValueError("near-zero/tolerance must be non-negative and max_iter positive")
        if self.split_half_repeats <= 0 or self.bootstrap_repeats <= 0:
            raise ValueError("stability repeat counts must be positive")
        for name in (
            "min_val_global_vaf",
            "min_val_local_vaf_quantile",
            "min_initialization_similarity",
            "min_split_half_similarity",
            "min_bootstrap_similarity",
            "min_cross_trial_similarity",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if not 0.0 <= float(self.local_vaf_quantile) <= 1.0:
            raise ValueError("local_vaf_quantile must lie in [0,1]")
        if isinstance(self.max_basis_condition_number, bool):
            raise ValueError("max_basis_condition_number must be finite and positive")
        max_condition = float(self.max_basis_condition_number)
        if not np.isfinite(max_condition) or max_condition <= 0.0:
            raise ValueError("max_basis_condition_number must be finite and positive")
        if isinstance(self.min_effective_rank_fraction, bool):
            raise ValueError("min_effective_rank_fraction must lie in [0,1]")
        effective_rank_fraction = float(self.min_effective_rank_fraction)
        if not 0.0 <= effective_rank_fraction <= 1.0:
            raise ValueError("min_effective_rank_fraction must lie in [0,1]")
        hybrid_config = _hybrid_config_from_fit(self).validated()
        weights = self.phase_weights or DEFAULT_PHASE_WEIGHTS
        phase_balanced_weights(np.asarray(sorted(weights), dtype=np.int32), weights=weights)
        return SynergyFitConfig(
            **{
                **asdict(self),
                "ranks": ranks,
                "region_ranks": region_ranks,
                "total_rank_budget": (
                    None if self.total_rank_budget is None else int(self.total_rank_budget)
                ),
                "expected_environment_fingerprint": expected_environment,
                "expected_rollout_manifest_fingerprint": expected_rollout,
                "max_basis_condition_number": max_condition,
                "min_effective_rank_fraction": effective_rank_fraction,
                "hybrid_novelty_residual_ratio": hybrid_config.novelty_residual_ratio,
                "hybrid_duplicate_cosine_similarity": hybrid_config.duplicate_cosine_similarity,
                "hybrid_min_heldout_global_vaf_marginal_gain": (
                    hybrid_config.min_heldout_global_vaf_marginal_gain
                ),
                "hybrid_max_total_rank": hybrid_config.max_total_rank,
                "hybrid_min_heldout_global_vaf": hybrid_config.min_heldout_global_vaf,
                "hybrid_local_vaf_quantile": hybrid_config.local_vaf_quantile,
                "hybrid_min_heldout_local_vaf_quantile": (
                    hybrid_config.min_heldout_local_vaf_quantile
                ),
                "hybrid_max_basis_condition_number": hybrid_config.max_basis_condition_number,
                "hybrid_min_effective_rank_fraction": hybrid_config.min_effective_rank_fraction,
                "hybrid_effective_rank_relative_tolerance": (
                    hybrid_config.effective_rank_relative_tolerance
                ),
                "seeds": seeds,
                "phase_weights": {int(key): float(value) for key, value in weights.items()},
            }
        )


def _hybrid_config_from_fit(config: SynergyFitConfig) -> HybridBasisConfig:
    return HybridBasisConfig(
        novelty_residual_ratio=config.hybrid_novelty_residual_ratio,
        duplicate_cosine_similarity=config.hybrid_duplicate_cosine_similarity,
        min_heldout_global_vaf_marginal_gain=(
            config.hybrid_min_heldout_global_vaf_marginal_gain
        ),
        max_total_rank=config.hybrid_max_total_rank,
        min_heldout_global_vaf=config.hybrid_min_heldout_global_vaf,
        local_vaf_quantile=config.hybrid_local_vaf_quantile,
        min_heldout_local_vaf_quantile=config.hybrid_min_heldout_local_vaf_quantile,
        max_basis_condition_number=config.hybrid_max_basis_condition_number,
        min_effective_rank_fraction=config.hybrid_min_effective_rank_fraction,
        effective_rank_relative_tolerance=(
            config.hybrid_effective_rank_relative_tolerance
        ),
    )


@dataclass(frozen=True)
class LoadedSynergySplit:
    source: Path
    split: str
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    muscle_names: tuple[str, ...]
    source_files: tuple[Path, ...]
    content_fingerprint: str
    motion_id_field: str | None

    @property
    def phase_id(self) -> np.ndarray:
        if "phase_id" not in self.arrays:
            raise ValueError(f"{self.split} synergy source has no event phase_id; frame-progress phase is not accepted")
        raw = np.asarray(self.arrays["phase_id"])
        if raw.ndim != 1:
            raise ValueError("phase_id must have shape [samples]")
        if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
            raw.dtype,
            np.integer,
        ):
            raise ValueError("phase_id must use an integer dtype; truncation is forbidden")
        return raw.astype(np.int32)

    @property
    def motion_ids(self) -> np.ndarray | None:
        if self.motion_id_field is None:
            return None
        raw = np.asarray(self.arrays[self.motion_id_field])
        if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
            raw.dtype,
            np.integer,
        ):
            raise ValueError(f"{self.motion_id_field} must use an integer dtype; truncation is forbidden")
        if np.issubdtype(raw.dtype, np.unsignedinteger) and np.any(raw > np.iinfo(np.int64).max):
            raise ValueError(f"{self.motion_id_field} exceeds signed int64 range")
        result = raw.astype(np.int64)
        if np.any(result < 0):
            raise ValueError(f"{self.motion_id_field} must be non-negative")
        return result

    def signal(self, signal_kind: str) -> SynergySignal:
        kind = _canonical_signal_kind(signal_kind)
        if kind == EXCITATION_SIGNAL_KIND:
            return _load_explicit_excitation(self)
        if "muscle_activation" not in self.arrays:
            raise ValueError(f"{self.split} source is missing muscle_activation")
        validate_physical_signal_semantics(self.metadata.get("physical_signal_semantics"))
        capture = self.metadata.get("physical_capture")
        if not isinstance(capture, Mapping) or capture.get("schema_version") != PHYSICAL_CAPTURE_SCHEMA_VERSION:
            raise ValueError(
                f"{self.split} activation fitting requires physical_capture_spec_v2 metadata; "
                "legacy datasets are rejected"
            )
        capture_names = tuple(str(name) for name in capture.get("actuator_names", ()))
        if capture_names != self.muscle_names:
            raise ValueError(f"{self.split} activation capture actuator order differs from dataset metadata")
        channel_contract = validate_muscle_channel_contract(
            capture.get("muscle_channel_contract"),
            expected_names=self.muscle_names,
        )
        valid = validate_activation_valid_mask(
            capture.get("activation_valid_mask"),
            expected_width=len(self.muscle_names),
        )
        if not np.all(valid):
            invalid = [self.muscle_names[index] for index in np.flatnonzero(~valid).tolist()]
            raise ValueError(
                f"{self.split} activation fitting includes actuators without scalar activation state: {invalid}"
            )
        transform = SignalTransform(
            kind="identity_nonnegative_activation",
            raw_signal_kind=MUSCLE_ACTIVATION_SOURCE,
            formula="activation",
            actuator_names=self.muscle_names,
            roundoff_policy=UNIT_INTERVAL_ROUNDOFF_POLICY,
            physical_signal_schema_version=PHYSICAL_SIGNAL_SCHEMA_VERSION,
            muscle_channel_contract=channel_contract.to_metadata(),
        )
        return SynergySignal(
            values=validate_unit_muscle_activation(self.arrays["muscle_activation"]),
            muscle_names=self.muscle_names,
            signal_kind=ACTIVATION_SIGNAL_KIND,
            transform=transform,
        ).validated()

    def provenance(self) -> dict[str, Any]:
        motion_ids = self.motion_ids
        return {
            "split": self.split,
            "source": str(self.source.resolve()),
            "source_files": [path.name for path in self.source_files],
            "source_file_sha256": {path.name: _file_sha256(path) for path in self.source_files},
            "content_fingerprint": self.content_fingerprint,
            "num_samples": int(self.phase_id.shape[0]),
            "motion_id_field": self.motion_id_field,
            "motion_uids": [] if motion_ids is None else [int(value) for value in np.unique(motion_ids).tolist()],
            "session_provenance": _jsonable(self.metadata.get("session_provenance")),
        }


def load_synergy_split(source: str | Path, *, split: str) -> LoadedSynergySplit:
    """Load one strict train/validation view from NPZ shard(s)."""

    path = Path(source)
    if path.is_dir():
        files = tuple(sorted(path.glob(f"{split}_*.npz")))
        if not files:
            files = tuple(sorted(path.glob("shard_*.npz")))
        if not files:
            raise FileNotFoundError(f"no {split}_*.npz or shard_*.npz files in {path}")
        metadata_path = path / "metadata.json"
    elif path.is_file() and path.suffix == ".npz":
        files = (path,)
        metadata_path = path.parent / "metadata.json"
    else:
        raise FileNotFoundError(f"synergy source does not exist or is not NPZ: {path}")

    metadata = load_json_strict(metadata_path) if metadata_path.is_file() else {}
    if not isinstance(metadata, dict):
        raise ValueError("dataset metadata.json must contain an object")
    arrays, embedded = _load_npz_fields(files)
    metadata = _merge_embedded_metadata(metadata, embedded)
    names = tuple(str(name) for name in metadata.get("actuator_names", ()))
    if not names or len(set(names)) != len(names):
        raise ValueError("synergy source requires unique ordered metadata.actuator_names")
    sample_count = int(next(iter(arrays.values())).shape[0]) if arrays else 0
    for field in ("teacher_ctrl_physical", "muscle_excitation", "muscle_activation"):
        if field in arrays and np.asarray(arrays[field]).shape != (sample_count, len(names)):
            raise ValueError(f"{field} must have shape [{sample_count},{len(names)}], got {arrays[field].shape}")
    if "phase_id" not in arrays:
        raise ValueError("synergy fitting requires event-derived phase_id in every shard")
    if np.asarray(arrays["phase_id"]).shape != (sample_count,):
        raise ValueError("phase_id must have shape [samples]")
    motion_field = "motion_uid" if "motion_uid" in arrays else ("traj_no" if "traj_no" in arrays else None)
    fingerprint = _source_fingerprint(files, metadata)
    return LoadedSynergySplit(
        source=path,
        split=str(split),
        arrays=arrays,
        metadata=metadata,
        muscle_names=names,
        source_files=files,
        content_fingerprint=fingerprint,
        motion_id_field=motion_field,
    )


def subset_signal(signal: SynergySignal, indices: Sequence[int]) -> SynergySignal:
    """Select an ordered region while preserving exact transform semantics."""

    source = signal.validated()
    selected = np.asarray(tuple(int(index) for index in indices), dtype=np.int32)
    if selected.ndim != 1 or selected.size == 0 or len(set(selected.tolist())) != selected.size:
        raise ValueError("regional indices must be a non-empty unique rank-1 sequence")
    if np.any(selected < 0) or np.any(selected >= len(source.muscle_names)):
        raise ValueError("regional indices are outside the muscle schema")
    names = tuple(source.muscle_names[int(index)] for index in selected)
    transform = source.transform
    if source.signal_kind == EXCITATION_SIGNAL_KIND:
        if transform is None or transform.ctrlrange is None:
            raise ValueError("excitation region lost its ctrlrange transform")
        limits = np.asarray(transform.ctrlrange, dtype=np.float64)[selected]
        transform = SignalTransform(
            kind=transform.kind,
            raw_signal_kind=transform.raw_signal_kind,
            formula=transform.formula,
            ctrlrange=limits,
            actuator_names=names,
            ctrlrange_schema_hash=ctrlrange_schema_hash(names, limits),
            roundoff_policy=transform.roundoff_policy,
            physical_signal_schema_version=transform.physical_signal_schema_version,
            muscle_channel_contract=validate_muscle_channel_contract(
                transform.muscle_channel_contract,
                expected_names=source.muscle_names,
            ).subset(selected.tolist()).to_metadata(),
        )
    elif transform is not None:
        channel_contract = validate_muscle_channel_contract(
            transform.muscle_channel_contract,
            expected_names=source.muscle_names,
        ).subset(selected.tolist())
        transform = SignalTransform(
            kind=transform.kind,
            raw_signal_kind=transform.raw_signal_kind,
            formula=transform.formula,
            actuator_names=names,
            roundoff_policy=transform.roundoff_policy,
            physical_signal_schema_version=transform.physical_signal_schema_version,
            muscle_channel_contract=channel_contract.to_metadata(),
        )
    return SynergySignal(
        values=source.values[:, selected],
        muscle_names=names,
        signal_kind=source.signal_kind,
        transform=transform,
    ).validated()


def fit_synergy_region(
    train_signal: SynergySignal,
    val_signal: SynergySignal,
    *,
    train_phase_id: np.ndarray,
    val_phase_id: np.ndarray,
    output_path: str | Path,
    region: str,
    teacher_checkpoint_fingerprint: str,
    source_dataset_fingerprint: str,
    split_provenance: Mapping[str, Any],
    config: SynergyFitConfig | None = None,
    train_motion_ids: np.ndarray | None = None,
    primitive_source_binding: Mapping[str, Any] | None = None,
    train_task_ids: np.ndarray | None = None,
    val_task_ids: np.ndarray | None = None,
    train_trial_ids: np.ndarray | None = None,
    val_trial_ids: np.ndarray | None = None,
    train_quality_weights: np.ndarray | None = None,
    val_quality_weights: np.ndarray | None = None,
    dynamic_coverage_reports: Mapping[int | str, Mapping[str, Any]] | None = None,
    defer_dynamic_coverage_selection: bool = False,
) -> dict[str, Any]:
    """Fit/rank/select one region and persist a decoder-ready basis artifact."""

    cfg = (config or SynergyFitConfig()).validated()
    train = train_signal.validated()
    val = val_signal.validated()
    _validate_signal_pair(train, val)
    checkpoint_fingerprint = str(teacher_checkpoint_fingerprint).strip()
    dataset_fingerprint = str(source_dataset_fingerprint).strip()
    if not checkpoint_fingerprint or not dataset_fingerprint:
        raise ValueError("teacher and source dataset fingerprints are required")
    train_phases = np.asarray(train_phase_id, dtype=np.int32)
    val_phases = np.asarray(val_phase_id, dtype=np.int32)
    if train_phases.shape != (train.values.shape[0],) or val_phases.shape != (val.values.shape[0],):
        raise ValueError("phase_id rows must match train/validation signal rows")

    train_processed, preprocess = fit_preprocessor(
        train.values,
        muscle_names=train.muscle_names,
        signal_kind=train.signal_kind,
        transform=train.transform,
        normalization=cfg.normalization,
        near_zero_threshold=cfg.near_zero_threshold,
    )
    val_processed = apply_preprocessor(
        val.values,
        preprocess,
        signal_kind=val.signal_kind,
        transform=val.transform,
    )
    if primitive_source_binding is not None:
        dropped = np.ones(train.values.shape[1], dtype=bool)
        dropped[preprocess.kept_indices] = False
        validation_rms = np.sqrt(np.mean(np.square(val.values), axis=0))
        newly_active = np.flatnonzero(dropped & (validation_rms > cfg.near_zero_threshold))
        if newly_active.size:
            names = [train.muscle_names[int(index)] for index in newly_active]
            raise BasisNotEligibleForEarlyControl(
                f"primitive validation activates muscles dropped as train-near-zero: {names}"
            )
    if primitive_source_binding is None:
        train_weights = phase_balanced_weights(train_phases, weights=cfg.phase_weights)
        val_weights = phase_balanced_weights(val_phases, weights=cfg.phase_weights)
        sample_balancing = {
            "kind": "phase_balanced",
            "quality_weighted": False,
        }
    else:
        train_weights = primitive_task_phase_balanced_weights(
            train_task_ids,
            train_phases,
            trial_ids=train_trial_ids,
            quality_weights=train_quality_weights,
            phase_weights=cfg.phase_weights,
        )
        val_weights = primitive_task_phase_balanced_weights(
            val_task_ids,
            val_phases,
            trial_ids=val_trial_ids,
            quality_weights=val_quality_weights,
            phase_weights=cfg.phase_weights,
        )
        sample_balancing = {
            "kind": "primitive_task_phase_trial_balanced",
            "quality_weighted": True,
            "formula": "equal_task_then_phase_weight_then_trial_mean_quality_then_frame_quality",
        }
    weighted_train = apply_sample_weights(train_processed, train_weights)
    weighted_val = apply_sample_weights(val_processed, val_weights)

    configured_ranks = candidate_ranks_for_region(
        cfg.ranks,
        cfg.region_ranks,
        region=str(region),
    )
    max_rank = min(weighted_train.shape)
    ranks = tuple(rank for rank in configured_ranks if rank <= max_rank)
    rejected_ranks = tuple(rank for rank in configured_ranks if rank > max_rank)
    if not ranks:
        raise BasisNotEligibleForEarlyControl(
            f"no candidate rank fits region {region!r} preprocessed matrix {weighted_train.shape}; "
            f"requested {configured_ranks}"
        )
    validate_dynamic_coverage_rank_inventory(
        dynamic_coverage_reports,
        candidate_ranks=ranks,
        label=f"dynamic coverage inventory for region {region!r}",
    )

    rank_reports: dict[int, dict[str, Any]] = {}
    best_results: dict[int, NMFResult] = {}
    candidate_inventory_entries: list[dict[str, Any]] = []
    transform_manifest = None if train.transform is None else train.transform.to_manifest()
    for rank in ranks:
        best, initializations = fit_best_initialization(
            weighted_train,
            rank=rank,
            seeds=cfg.seeds,
            max_iter=cfg.max_iter,
            tol=cfg.tol,
        )
        best_results[rank] = best
        train_metrics = _evaluate_basis(train_processed, best.basis)
        val_metrics = _evaluate_basis(val_processed, best.basis)
        train_balanced_metrics = _evaluate_basis(weighted_train, best.basis)
        val_balanced_metrics = _evaluate_basis(weighted_val, best.basis)
        init_report = initialization_stability(initializations)
        split_report = split_half_stability(
            weighted_train,
            rank=rank,
            repeats=cfg.split_half_repeats,
            seed=cfg.seeds[0] + 10_000 + rank * 100,
            max_iter=cfg.max_iter,
        )
        bootstrap_report = bootstrap_stability(
            weighted_train,
            reference_basis=best.basis,
            rank=rank,
            repeats=cfg.bootstrap_repeats,
            seed=cfg.seeds[0] + 20_000 + rank * 100,
            max_iter=cfg.max_iter,
        )
        cross_trial_report = _fit_cross_trial_stability(
            weighted_train,
            train_motion_ids if primitive_source_binding is None else train_trial_ids,
            task_ids=(None if primitive_source_binding is None else train_task_ids),
            rank=rank,
            seed=cfg.seeds[0] + 30_000 + rank * 100,
            max_iter=cfg.max_iter,
            max_trials=cfg.cross_trial_max_trials,
        )
        eligibility_metrics = val_balanced_metrics if primitive_source_binding is not None else val_metrics
        local_quantile = _finite_quantile(
            eligibility_metrics["local_vaf"],
            cfg.local_vaf_quantile,
        )
        primitive_group_validation = (
            None
            if primitive_source_binding is None
            else _primitive_group_validation_metrics(
                val_processed,
                best.basis,
                task_ids=val_task_ids,
                phase_ids=val_phases,
                trial_ids=val_trial_ids,
            )
        )
        physical_candidate = np.zeros((len(train.muscle_names), rank), dtype=np.float64)
        physical_candidate[preprocess.kept_indices] = preprocess.scales[:, None] * best.basis
        persisted_physical_candidate = physical_candidate.astype(np.float32).astype(
            np.float64
        )
        physical_condition_number = basis_condition_number(persisted_physical_candidate)
        physical_effective_rank_fraction = _basis_effective_rank_fraction(
            persisted_physical_candidate
        )
        rejection_reasons = _rank_rejection_reasons(
            val_global_vaf=float(eligibility_metrics["global_vaf"]),
            val_local_quantile=local_quantile,
            initialization=float(init_report["mean_similarity"]),
            split_half=float(split_report["mean_similarity"]),
            bootstrap=float(bootstrap_report["mean_similarity"]),
            cross_trial=cross_trial_report,
            primitive_group_min_vaf=(
                None
                if primitive_group_validation is None
                else float(primitive_group_validation["minimum_global_vaf"])
            ),
            basis_condition_number_value=physical_condition_number,
            effective_rank_fraction=physical_effective_rank_fraction,
            config=cfg,
        )
        offline_rejection_reasons = list(rejection_reasons)
        candidate_fingerprint = candidate_basis_fingerprint(
            physical_candidate,
            muscle_names=train.muscle_names,
            signal_kind=train.signal_kind,
            region=str(region),
        )
        if cfg.require_dynamic_coverage:
            candidate_path = Path(output_path) / "candidates" / f"rank_{rank:04d}"
            candidate_artifact = save_synergy_basis(
                candidate_path,
                basis=physical_candidate,
                muscle_names=train.muscle_names,
                manifest=_jsonable(
                    {
                        "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
                        "signal_kind": train.signal_kind,
                        "region": str(region),
                        "rank": rank,
                        "normalization": preprocess.to_manifest(),
                        "basis_space": "source_signal_units_after_train_only_denormalization",
                        "source_dataset_fingerprint": dataset_fingerprint,
                        "teacher_checkpoint_fingerprint": checkpoint_fingerprint,
                        "fit_seed": best.seed,
                        "transform": transform_manifest,
                        "split_provenance": split_provenance,
                        "train_motion_uids": _unique_ints(train_motion_ids),
                        "primitive_source_binding": primitive_source_binding,
                        "candidate_role": "dynamic_coverage_rollout_candidate",
                        "candidate_basis_fingerprint": candidate_fingerprint,
                        "fit_config": asdict(cfg),
                    }
                ),
            )
            persisted_candidate_fingerprint = candidate_basis_fingerprint(
                candidate_artifact.basis,
                muscle_names=candidate_artifact.muscle_names,
                signal_kind=train.signal_kind,
                region=str(region),
            )
            if persisted_candidate_fingerprint != candidate_fingerprint:
                raise RuntimeError(
                    "persisted dynamic-coverage candidate differs from its bound fingerprint"
                )
            candidate_inventory_entries.append(
                {
                    "rank": rank,
                    "candidate_basis_fingerprint": candidate_fingerprint,
                    "candidate_artifact_fingerprint": candidate_artifact.fingerprint,
                    "candidate_artifact_path": str(
                        Path("candidates") / f"rank_{rank:04d}"
                    ),
                    "offline_eligible": not offline_rejection_reasons,
                    "offline_rejection_reasons": offline_rejection_reasons,
                }
            )
        dynamic_report = dynamic_coverage_report_for_rank(
            dynamic_coverage_reports,
            rank=rank,
        )
        validated_dynamic: dict[str, Any] | None = None
        dynamic_validation_error: str | None = None
        if dynamic_report is not None:
            try:
                validated_dynamic = validate_dynamic_coverage_gate(
                    dynamic_report,
                    region=str(region),
                    rank=rank,
                    candidate_fingerprint=candidate_fingerprint,
                    signal_kind=train.signal_kind,
                    max_mean_dynamic_gap=cfg.max_mean_dynamic_gap,
                    max_key_phase_dynamic_gap=cfg.max_key_phase_dynamic_gap,
                    expected_environment_fingerprint=cfg.expected_environment_fingerprint,
                    expected_rollout_manifest_fingerprint=(
                        cfg.expected_rollout_manifest_fingerprint
                    ),
                )
            except (TypeError, ValueError) as exc:
                dynamic_validation_error = str(exc)
        if cfg.require_dynamic_coverage:
            if dynamic_report is None:
                rejection_reasons.append("required_dynamic_coverage_evidence_missing")
            elif dynamic_validation_error is not None:
                rejection_reasons.append("required_dynamic_coverage_evidence_invalid")
            elif validated_dynamic is None or validated_dynamic["passed"] is not True:
                rejection_reasons.append("required_dynamic_coverage_gate_failed")
        rank_reports[rank] = {
            "rank": rank,
            "best_seed": best.seed,
            "best_weighted_loss": best.loss,
            "best_n_iter": best.n_iter,
            "initialization_losses": {str(item.seed): item.loss for item in initializations},
            "train": train_metrics,
            "validation": val_metrics,
            "train_phase_balanced": train_balanced_metrics,
            "validation_phase_balanced": val_balanced_metrics,
            "validation_local_vaf_quantile": local_quantile,
            "validation_local_vaf_quantile_level": cfg.local_vaf_quantile,
            "initialization_stability": init_report,
            "split_half_stability": split_report,
            "bootstrap_stability": bootstrap_report,
            "cross_trial_stability": cross_trial_report,
            "primitive_group_validation": primitive_group_validation,
            "candidate_basis_fingerprint": candidate_fingerprint,
            "numerical_conditioning": {
                "basis_condition_number": physical_condition_number,
                "effective_rank_fraction": physical_effective_rank_fraction,
            },
            "offline_eligible": not offline_rejection_reasons,
            "offline_rejection_reasons": offline_rejection_reasons,
            "dynamic_coverage_required": cfg.require_dynamic_coverage,
            "dynamic_coverage": validated_dynamic,
            "dynamic_coverage_validation_error": dynamic_validation_error,
            "eligible": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
        }

    candidate_inventory_path: Path | None = None
    candidate_inventory_fingerprint: str | None = None
    if cfg.require_dynamic_coverage:
        candidate_inventory_path = Path(output_path) / "candidate_inventory.json"
        candidate_inventory_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_inventory = {
            "schema_version": "synergy_dynamic_coverage_candidate_inventory_v1",
            "signal_kind": train.signal_kind,
            "region": str(region),
            "source_dataset_fingerprint": dataset_fingerprint,
            "teacher_checkpoint_fingerprint": checkpoint_fingerprint,
            "expected_environment_fingerprint": cfg.expected_environment_fingerprint,
            "expected_rollout_manifest_fingerprint": (
                cfg.expected_rollout_manifest_fingerprint
            ),
            "thresholds": {
                "max_mean_dynamic_gap": cfg.max_mean_dynamic_gap,
                "max_key_phase_dynamic_gap": cfg.max_key_phase_dynamic_gap,
            },
            "candidate_ranks": list(ranks),
            "rejected_too_large_ranks": list(rejected_ranks),
            "candidates": candidate_inventory_entries,
        }
        candidate_inventory_fingerprint = _json_sha256(candidate_inventory)
        candidate_inventory["inventory_fingerprint"] = candidate_inventory_fingerprint
        candidate_inventory_path.write_text(
            json.dumps(
                _jsonable(candidate_inventory),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    unresolved_dynamic_ranks = [
        rank
        for rank in ranks
        if rank_reports[rank]["offline_eligible"]
        and (
            "required_dynamic_coverage_evidence_missing"
            in rank_reports[rank]["rejection_reasons"]
            or "required_dynamic_coverage_evidence_invalid"
            in rank_reports[rank]["rejection_reasons"]
        )
    ]
    if cfg.require_dynamic_coverage and unresolved_dynamic_ranks:
        pending = {
            "status": "dynamic_coverage_evidence_required",
            "region": str(region),
            "signal_kind": train.signal_kind,
            "candidate_inventory_path": str(candidate_inventory_path.resolve()),
            "candidate_inventory_fingerprint": candidate_inventory_fingerprint,
            "unresolved_offline_eligible_ranks": unresolved_dynamic_ranks,
            "rank_scan": {str(rank): report for rank, report in rank_reports.items()},
        }
        if defer_dynamic_coverage_selection:
            return pending
        raise BasisNotEligibleForEarlyControl(
            "dynamic coverage requires second-stage environment-rollout evidence for "
            f"offline-eligible ranks {unresolved_dynamic_ranks}; candidate inventory: "
            f"{candidate_inventory_path.resolve()}"
        )

    selected_rank = select_smallest_eligible_rank(rank_reports, region=str(region))
    eligible = [rank for rank in ranks if rank_reports[rank]["eligible"]]
    # Keep the established action-interface ABI string.  The optional dynamic
    # gate is represented explicitly in selected_metrics/selection.dynamic_coverage_gate.
    selection_reason = "smallest_rank_meeting_all_vaf_and_stability_gates"
    selected = best_results[selected_rank]

    # NMF is fitted in normalized coordinates.  Undo only the channel scaling
    # and restore removed near-zero rows so the saved W consumes the complete,
    # ordered physical signal schema expected by a decoder.
    physical_kept_basis = preprocess.scales[:, None] * selected.basis
    physical_basis = np.zeros((len(train.muscle_names), selected_rank), dtype=np.float64)
    physical_basis[preprocess.kept_indices] = physical_kept_basis
    selected_report = rank_reports[selected_rank]
    manifest = {
        "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
        "signal_kind": train.signal_kind,
        "region": str(region),
        "rank": selected_rank,
        "normalization": preprocess.to_manifest(),
        "basis_space": "source_signal_units_after_train_only_denormalization",
        "source_dataset_fingerprint": dataset_fingerprint,
        "teacher_checkpoint_fingerprint": checkpoint_fingerprint,
        "fit_seed": selected.seed,
        "transform": transform_manifest,
        "split_provenance": _jsonable(split_provenance),
        "train_motion_uids": _unique_ints(train_motion_ids),
        "primitive_source_binding": _jsonable(primitive_source_binding),
        "phase_balancing": {
            "weights": {str(key): float(value) for key, value in cfg.phase_weights.items()},
            "sample_weight_application": "multiply_rows_by_sqrt_weight",
            "fit_scope": "train_only",
            "sample_balancing": sample_balancing,
        },
        "selection": {
            "selected_rank": selected_rank,
            "reason": selection_reason,
            "eligible_ranks": eligible,
            "rejected_too_large_ranks": list(rejected_ranks),
            "thresholds": {
                "min_val_global_vaf": cfg.min_val_global_vaf,
                "min_val_local_vaf_quantile": cfg.min_val_local_vaf_quantile,
                "local_vaf_quantile": cfg.local_vaf_quantile,
                "min_initialization_similarity": cfg.min_initialization_similarity,
                "min_split_half_similarity": cfg.min_split_half_similarity,
                "min_bootstrap_similarity": cfg.min_bootstrap_similarity,
                "min_cross_trial_similarity": cfg.min_cross_trial_similarity,
                "max_basis_condition_number": cfg.max_basis_condition_number,
                "min_effective_rank_fraction": cfg.min_effective_rank_fraction,
            },
            "dynamic_coverage_gate": dynamic_coverage_requirement(
                required=cfg.require_dynamic_coverage,
                max_mean_dynamic_gap=cfg.max_mean_dynamic_gap,
                max_key_phase_dynamic_gap=cfg.max_key_phase_dynamic_gap,
                expected_environment_fingerprint=cfg.expected_environment_fingerprint,
                expected_rollout_manifest_fingerprint=(
                    cfg.expected_rollout_manifest_fingerprint
                ),
            ),
        },
        "selected_metrics": selected_report,
        "rank_scan": {str(rank): report for rank, report in rank_reports.items()},
        "fit_config": asdict(cfg),
    }
    artifact = save_synergy_basis(
        output_path,
        basis=physical_basis,
        muscle_names=train.muscle_names,
        manifest=_jsonable(manifest),
    )
    # Coefficient limits used by an early Stage-1 action wrapper must describe
    # the final decoder-ready physical W on the unweighted training samples.
    # The internally fitted coefficients belong to phase-weighted, normalized
    # rows and therefore cannot be reused for this action contract.
    physical_coefficients, _ = transform_nmf(train.values, artifact.basis)
    coefficient_stats = save_coefficient_statistics(
        artifact.path / "coefficient_stats.npz",
        physical_coefficients,
        basis_fingerprint=artifact.fingerprint,
    )
    return {
        "region": str(region),
        "signal_kind": train.signal_kind,
        "selected_rank": selected_rank,
        "selection_reason": selection_reason,
        "artifact_path": str(artifact.path.resolve()),
        "artifact_fingerprint": artifact.fingerprint,
        "coefficient_statistics_path": coefficient_stats["path"],
        "coefficient_statistics_fingerprint": coefficient_stats["stats_fingerprint"],
        "selected_metrics": _jsonable(selected_report),
        "candidate_ranks": list(ranks),
        "rejected_too_large_ranks": list(rejected_ranks),
    }


def fit_synergy_dataset(
    train_source: str | Path,
    val_source: str | Path,
    *,
    output_dir: str | Path,
    teacher_checkpoint_fingerprint: str | None = None,
    signal_kinds: Sequence[str] = (EXCITATION_SIGNAL_KIND, ACTIVATION_SIGNAL_KIND),
    mode: str = "both",
    grouping_json: str | Path | None = None,
    anatomical_taxonomy_json: str | Path | None = None,
    primitive_source_manifest: str | Path | None = None,
    config: SynergyFitConfig | None = None,
    dynamic_coverage_reports: Mapping[
        str,
        Mapping[str, Mapping[int | str, Mapping[str, Any]]],
    ]
    | None = None,
) -> dict[str, Any]:
    """Fit global/regional excitation and activation artifacts from disk."""

    cfg = (config or SynergyFitConfig()).validated()
    train = load_synergy_split(train_source, split="train")
    val = load_synergy_split(val_source, split="val")
    if train.muscle_names != val.muscle_names:
        raise ValueError("train/validation ordered actuator names differ")
    if set(train.source_files) & set(val.source_files):
        raise ValueError("train/validation sources overlap at the shard-file level")
    if train.motion_ids is None or val.motion_ids is None:
        raise ValueError("synergy fitting requires stable motion_uid/traj_no identities in both splits")
    if primitive_source_manifest is not None and (
        train.motion_id_field != "motion_uid" or val.motion_id_field != "motion_uid"
    ):
        raise ValueError("primitive fitting requires stable motion_uid, never local traj_no")
    train_motion_uids = set(_unique_ints(train.motion_ids))
    val_motion_uids = set(_unique_ints(val.motion_ids))
    if not train_motion_uids or not val_motion_uids:
        raise ValueError("synergy train/validation motion identity sets must be non-empty")
    motion_overlap = sorted(train_motion_uids & val_motion_uids)
    if motion_overlap:
        raise ValueError(f"synergy train/validation motion leakage detected: {motion_overlap}")
    if primitive_source_manifest is None:
        checkpoint = _resolve_teacher_fingerprint(
            teacher_checkpoint_fingerprint,
            train.metadata,
            val.metadata,
        )
    else:
        primitive_preview = load_primitive_source_manifest(primitive_source_manifest)
        checkpoint = _json_sha256(
            {
                "schema_version": "primitive_checkpoint_inventory_v1",
                "source_checkpoint_fingerprints": primitive_preview.manifest["source_checkpoint_fingerprints"],
            }
        )
        if teacher_checkpoint_fingerprint is not None and str(teacher_checkpoint_fingerprint) != checkpoint:
            raise ValueError(
                "explicit teacher checkpoint fingerprint differs from the primitive checkpoint inventory aggregate"
            )
    mode = str(mode)
    if mode not in {"global", "regional", "both"}:
        raise ValueError("mode must be global, regional, or both")
    groups: dict[str, tuple[int, ...]] = {}
    regional: dict[str, tuple[int, ...]] = {}
    if mode in {"global", "both"}:
        groups.update(global_group(train.muscle_names))
    if mode in {"regional", "both"}:
        if grouping_json is None:
            raise ValueError("regional fitting requires --grouping-json with explicit muscle ownership")
        taxonomy = None
        if anatomical_taxonomy_json is not None:
            from musclemimic.physiology import load_anatomical_taxonomy

            taxonomy = load_anatomical_taxonomy(anatomical_taxonomy_json)
        regional = load_grouping_json(
            grouping_json,
            muscle_names=train.muscle_names,
            require_complete=True,
            taxonomy=taxonomy,
        )
        duplicate = sorted(set(groups) & set(regional))
        if duplicate:
            raise ValueError(f"regional labels collide with global labels: {duplicate}")
        groups.update(regional)

    unknown_rank_regions = set(cfg.region_ranks or {}) - set(groups)
    if unknown_rank_regions:
        raise ValueError(
            "region_ranks contains labels absent from the requested grouping: "
            f"{sorted(unknown_rank_regions)}"
        )

    canonical_signal_kinds = tuple(_canonical_signal_kind(value) for value in signal_kinds)
    configured_region_ranks = {
        region: candidate_ranks_for_region(cfg.ranks, cfg.region_ranks, region=region)
        for region in groups
    }
    _validate_dynamic_coverage_inventory(
        dynamic_coverage_reports,
        signal_kinds=canonical_signal_kinds,
        region_candidate_ranks=configured_region_ranks,
        allow_hybrid=(mode == "both"),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _invalidate_output_file(output / "fit_report.json")
    _invalidate_output_file(output / "promotion_metrics.json")
    if mode == "both":
        for kind in canonical_signal_kinds:
            _invalidate_hybrid_primary_artifact(
                output / kind / "hybrid_global_regional"
            )
    split_provenance = {"train": train.provenance(), "validation": val.provenance()}
    combined_dataset_fingerprint = _json_sha256(split_provenance)
    primitive_source_binding = _load_primitive_source_binding(
        primitive_source_manifest,
        source_dataset_fingerprint=combined_dataset_fingerprint,
        train_motion_uids=train_motion_uids,
        validation_motion_uids=val_motion_uids,
        actuator_names=train.muscle_names,
        train_metadata=train.metadata,
        validation_metadata=val.metadata,
        train_arrays=train.arrays,
        validation_arrays=val.arrays,
        fit_config=cfg,
    )
    reports: list[dict[str, Any]] = []
    preferred_decoder_artifacts: dict[str, dict[str, Any]] = {}
    pending_dynamic_regions: list[dict[str, Any]] = []
    for kind in canonical_signal_kinds:
        train_signal = train.signal(kind)
        val_signal = val.signal(kind)
        _validate_signal_pair(train_signal, val_signal)
        signal_reports: dict[str, dict[str, Any]] = {}
        for region, indices in groups.items():
            region_train = subset_signal(train_signal, indices)
            region_val = subset_signal(val_signal, indices)
            artifact_dir = output / kind / _safe_slug(region)
            fitted = fit_synergy_region(
                region_train,
                region_val,
                train_phase_id=train.phase_id,
                val_phase_id=val.phase_id,
                output_path=artifact_dir,
                region=region,
                teacher_checkpoint_fingerprint=checkpoint,
                source_dataset_fingerprint=combined_dataset_fingerprint,
                split_provenance=split_provenance,
                config=cfg,
                train_motion_ids=train.motion_ids,
                primitive_source_binding=primitive_source_binding,
                train_task_ids=(None if primitive_source_binding is None else np.asarray(train.arrays["task_id"])),
                val_task_ids=(None if primitive_source_binding is None else np.asarray(val.arrays["task_id"])),
                train_trial_ids=(None if primitive_source_binding is None else np.asarray(train.arrays["trial_id"])),
                val_trial_ids=(None if primitive_source_binding is None else np.asarray(val.arrays["trial_id"])),
                train_quality_weights=(
                    None
                    if primitive_source_binding is None
                    else np.asarray(train.arrays["quality_weight"], dtype=np.float64)
                ),
                val_quality_weights=(
                    None
                    if primitive_source_binding is None
                    else np.asarray(val.arrays["quality_weight"], dtype=np.float64)
                ),
                dynamic_coverage_reports=_dynamic_coverage_reports_for_region(
                    dynamic_coverage_reports,
                    signal_kind=kind,
                    region=region,
                ),
                defer_dynamic_coverage_selection=True,
            )
            if fitted.get("status") == "dynamic_coverage_evidence_required":
                pending_dynamic_regions.append(fitted)
                continue
            fitted["artifact_role"] = "global_comparator" if region == "whole_body" else "regional_component"
            reports.append(fitted)
            signal_reports[region] = fitted
        if any(item["signal_kind"] == kind for item in pending_dynamic_regions):
            continue
        if regional:
            component_reports = {region: signal_reports[region] for region in regional}
            composite = build_regional_composite_artifact(
                component_reports,
                full_signal=train_signal,
                validation_signal=val_signal,
                groups=regional,
                output_path=output / kind / "regional_composite",
                teacher_checkpoint_fingerprint=checkpoint,
                source_dataset_fingerprint=combined_dataset_fingerprint,
                split_provenance=split_provenance,
                train_motion_ids=train.motion_ids,
                min_local_vaf_coverage=cfg.min_val_local_vaf_quantile,
                primitive_source_binding=primitive_source_binding,
                total_rank_budget=cfg.total_rank_budget,
            )
            if mode == "both":
                composite["artifact_role"] = "regional_composite_source"
            reports.append(composite)
            if mode == "both":
                hybrid = build_hybrid_global_regional_artifact(
                    regional_composite_report=composite,
                    global_report=signal_reports["whole_body"],
                    train_signal=train_signal,
                    validation_signal=val_signal,
                    output_path=output / kind / "hybrid_global_regional",
                    teacher_checkpoint_fingerprint=checkpoint,
                    source_dataset_fingerprint=combined_dataset_fingerprint,
                    split_provenance=split_provenance,
                    train_motion_ids=train.motion_ids,
                    primitive_source_binding=primitive_source_binding,
                    config=cfg,
                    dynamic_coverage_reports=_dynamic_coverage_reports_for_region(
                        dynamic_coverage_reports,
                        signal_kind=kind,
                        region="hybrid_global_regional",
                    ),
                )
                if hybrid.get("status") == "dynamic_coverage_evidence_required":
                    pending_dynamic_regions.append(hybrid)
                    continue
                reports.append(hybrid)
                preferred_decoder_artifacts[kind] = {
                    "artifact_path": hybrid["artifact_path"],
                    "artifact_fingerprint": hybrid["artifact_fingerprint"],
                    "reason": "qualified_hybrid_global_regional_is_primary_decoder_basis",
                }
            else:
                preferred_decoder_artifacts[kind] = {
                    "artifact_path": composite["artifact_path"],
                    "artifact_fingerprint": composite["artifact_fingerprint"],
                    "reason": "regional-only mode requested",
                }
        elif "whole_body" in signal_reports:
            preferred_decoder_artifacts[kind] = {
                "artifact_path": signal_reports["whole_body"]["artifact_path"],
                "artifact_fingerprint": signal_reports["whole_body"]["artifact_fingerprint"],
                "reason": "global-only mode requested",
            }
    if pending_dynamic_regions:
        inventory_path = output / "dynamic_coverage_candidate_inventory.json"
        inventory = {
            "schema_version": "synergy_dynamic_coverage_dataset_candidate_inventory_v1",
            "status": "dynamic_coverage_evidence_required",
            "source_dataset_fingerprint": combined_dataset_fingerprint,
            "teacher_checkpoint_fingerprint": checkpoint,
            "expected_environment_fingerprint": cfg.expected_environment_fingerprint,
            "expected_rollout_manifest_fingerprint": (
                cfg.expected_rollout_manifest_fingerprint
            ),
            "thresholds": {
                "max_mean_dynamic_gap": cfg.max_mean_dynamic_gap,
                "max_key_phase_dynamic_gap": cfg.max_key_phase_dynamic_gap,
            },
            "regions": [
                {
                    "signal_kind": item["signal_kind"],
                    "region": item["region"],
                    "candidate_inventory_path": str(
                        Path(item["candidate_inventory_path"])
                        .resolve()
                        .relative_to(output.resolve())
                    ),
                    "candidate_inventory_fingerprint": item[
                        "candidate_inventory_fingerprint"
                    ],
                    "unresolved_offline_eligible_ranks": item[
                        "unresolved_offline_eligible_ranks"
                    ],
                }
                for item in pending_dynamic_regions
            ],
        }
        inventory_fingerprint = _json_sha256(inventory)
        inventory["inventory_fingerprint"] = inventory_fingerprint
        inventory_path.write_text(
            json.dumps(
                _jsonable(inventory),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        pending_report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "dynamic_coverage_evidence_required",
            "source_dataset_fingerprint": combined_dataset_fingerprint,
            "teacher_checkpoint_fingerprint": checkpoint,
            "fit_config": asdict(cfg),
            "candidate_inventory_path": str(inventory_path.resolve()),
            "candidate_inventory_fingerprint": inventory_fingerprint,
        }
        (output / "fit_report.json").write_text(
            json.dumps(
                _jsonable(pending_report),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        raise BasisNotEligibleForEarlyControl(
            "dynamic coverage requires second-stage environment-rollout evidence; "
            f"candidate inventory: {inventory_path.resolve()}"
        )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "train_source": str(Path(train_source).resolve()),
        "validation_source": str(Path(val_source).resolve()),
        "teacher_checkpoint_fingerprint": checkpoint,
        "source_dataset_fingerprint": combined_dataset_fingerprint,
        "split_provenance": split_provenance,
        "primitive_source_binding": primitive_source_binding,
        "mode": mode,
        "grouping_json": None if grouping_json is None else str(Path(grouping_json).resolve()),
        "anatomical_taxonomy_json": (
            None if anatomical_taxonomy_json is None else str(Path(anatomical_taxonomy_json).resolve())
        ),
        "grouping_bound_to_anatomical_taxonomy": anatomical_taxonomy_json is not None,
        "signal_kinds": list(canonical_signal_kinds),
        "fit_config": asdict(cfg),
        "dynamic_coverage_evidence_supplied": dynamic_coverage_reports is not None,
        "preferred_decoder_artifacts": preferred_decoder_artifacts,
        "artifacts": reports,
    }
    excitation_primary = next(
        (
            item
            for item in reports
            if item["signal_kind"] == EXCITATION_SIGNAL_KIND
            and item["artifact_role"]
            in {"primary_hybrid_global_regional", "primary_regional_composite"}
        ),
        None,
    )
    promotion_metrics = (
        {
            "schema_version": "forehand_clear_synergy_promotion_metrics_v1",
            "heldout_sample_count": 0,
            "explained_variance": 0.0,
            "reconstruction_nrmse": 1.0,
            "muscle_coverage": 0.0,
            "basis_binding_verified": 0.0,
            "failure_reason": "no physical-excitation regional composite artifact was fitted",
        }
        if excitation_primary is None
        else excitation_primary["promotion_metrics"]
    )
    promotion_path = output / "promotion_metrics.json"
    promotion_path.write_text(
        json.dumps(
            _jsonable(promotion_metrics),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report["promotion_metrics_path"] = str(promotion_path.resolve())
    report_path = output / "fit_report.json"
    report_path.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_regional_composite_artifact(
    component_reports: Mapping[str, Mapping[str, Any]],
    *,
    full_signal: SynergySignal,
    validation_signal: SynergySignal,
    groups: Mapping[str, Sequence[int]],
    output_path: str | Path,
    teacher_checkpoint_fingerprint: str,
    source_dataset_fingerprint: str,
    split_provenance: Mapping[str, Any],
    train_motion_ids: np.ndarray | None,
    min_local_vaf_coverage: float = 0.70,
    primitive_source_binding: Mapping[str, Any] | None = None,
    total_rank_budget: int | None = None,
) -> dict[str, Any]:
    """Combine regional W blocks into one full-action decoder artifact."""

    signal = full_signal.validated()
    validation = validation_signal.validated()
    _validate_signal_pair(signal, validation)
    if list(component_reports) != list(groups):
        raise ValueError("regional component reports must exactly follow grouping order")
    total_rank = 0
    loaded_components = []
    descriptors: list[dict[str, Any]] = []
    normalizations: dict[str, Any] = {}
    component_fit_seeds: dict[str, int] = {}
    component_ranks: dict[str, int] = {}
    for region, raw_indices in groups.items():
        indices = tuple(int(index) for index in raw_indices)
        report = component_reports[region]
        artifact = load_synergy_basis(str(report["artifact_path"]))
        expected_names = tuple(signal.muscle_names[index] for index in indices)
        if artifact.muscle_names != expected_names:
            raise ValueError(f"regional artifact {region!r} names differ from grouping ownership")
        manifest = artifact.manifest
        if (
            manifest.get("region") != region
            or manifest.get("signal_kind") != signal.signal_kind
            or manifest.get("teacher_checkpoint_fingerprint") != teacher_checkpoint_fingerprint
            or manifest.get("source_dataset_fingerprint") != source_dataset_fingerprint
            or _json_sha256(manifest.get("split_provenance")) != _json_sha256(split_provenance)
            or _json_sha256(manifest.get("primitive_source_binding")) != _json_sha256(primitive_source_binding)
        ):
            raise ValueError(f"regional artifact {region!r} provenance differs from composite contract")
        rank = int(manifest["rank"])
        start = total_rank
        stop = start + rank
        descriptors.append(
            {
                "region": region,
                "row_indices": list(indices),
                "muscle_names": list(expected_names),
                "rank": rank,
                "column_start": start,
                "column_stop": stop,
                "component_artifact_fingerprint": artifact.fingerprint,
            }
        )
        normalizations[region] = manifest["normalization"]
        component_fit_seeds[region] = int(manifest["fit_seed"])
        component_ranks[region] = rank
        loaded_components.append((indices, start, stop, artifact))
        total_rank = stop
    if total_rank <= 0:
        raise ValueError("regional composite requires at least one fitted component")
    checked_total_rank = enforce_total_rank_budget(
        component_ranks,
        total_rank_budget=total_rank_budget,
    )
    if checked_total_rank != total_rank:
        raise ValueError("regional composite rank accounting is inconsistent")
    composite_basis = np.zeros((len(signal.muscle_names), total_rank), dtype=np.float64)
    for indices, start, stop, artifact in loaded_components:
        composite_basis[np.asarray(indices, dtype=np.int32), start:stop] = artifact.basis
    grouping_payload = {
        "full_muscle_names": list(signal.muscle_names),
        "regions": [
            {
                "region": item["region"],
                "row_indices": item["row_indices"],
                "muscle_names": item["muscle_names"],
            }
            for item in descriptors
        ],
    }
    manifest = {
        "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
        "signal_kind": signal.signal_kind,
        "region": "regional_composite",
        "rank": total_rank,
        "normalization": {
            "kind": "per_region_train_only",
            "regions": normalizations,
            "basis_space": "source_signal_units_after_train_only_denormalization",
        },
        "basis_space": "source_signal_units_block_diagonal_regional_composite",
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "teacher_checkpoint_fingerprint": teacher_checkpoint_fingerprint,
        "fit_seed": -1,
        "component_fit_seeds": component_fit_seeds,
        "transform": None if signal.transform is None else signal.transform.to_manifest(),
        "split_provenance": _jsonable(split_provenance),
        "train_motion_uids": _unique_ints(train_motion_ids),
        "primitive_source_binding": _jsonable(primitive_source_binding),
        "composite_schema_version": "regional_synergy_composite_v1",
        "total_rank_budget": total_rank_budget,
        "total_rank_budget_required": total_rank_budget is not None,
        "total_rank_budget_passed": None if total_rank_budget is None else True,
        "composite_regions": descriptors,
        "regional_grouping_fingerprint": _json_sha256(grouping_payload),
        "component_artifacts": {
            region: {
                "artifact_path": str(Path(component_reports[region]["artifact_path"]).resolve()),
                "artifact_fingerprint": component_reports[region]["artifact_fingerprint"],
            }
            for region in groups
        },
    }
    artifact = save_synergy_basis(
        output_path,
        basis=composite_basis,
        muscle_names=signal.muscle_names,
        manifest=_jsonable(manifest),
    )
    physical_coefficients, _ = transform_nmf(signal.values, artifact.basis)
    coefficient_stats = save_coefficient_statistics(
        artifact.path / "coefficient_stats.npz",
        physical_coefficients,
        basis_fingerprint=artifact.fingerprint,
    )
    _, heldout_reconstruction = transform_nmf(validation.values, artifact.basis)
    heldout_metrics = reconstruction_metrics(validation.values, heldout_reconstruction)
    signal_rms = float(np.sqrt(np.mean(np.square(validation.values))))
    reconstruction_nrmse = float(heldout_metrics["rmse"]) / signal_rms if signal_rms > 1e-12 else float("inf")
    local_vaf = np.asarray(heldout_metrics["local_vaf"], dtype=np.float64)
    active = np.sum(np.square(validation.values), axis=0) > 1e-12
    covered = active & np.isfinite(local_vaf) & (local_vaf >= float(min_local_vaf_coverage))
    muscle_coverage = float(np.mean(covered[active])) if np.any(active) else 0.0
    all_regions_eligible = all(bool(component_reports[region]["selected_metrics"]["eligible"]) for region in groups)
    binding_verified = load_synergy_basis(artifact.path).fingerprint == artifact.fingerprint and all(
        descriptor["component_artifact_fingerprint"] == component_reports[descriptor["region"]]["artifact_fingerprint"]
        for descriptor in descriptors
    )
    promotion_metrics = {
        "schema_version": "forehand_clear_synergy_promotion_metrics_v1",
        "heldout_sample_count": int(validation.values.shape[0]),
        "explained_variance": float(heldout_metrics["global_vaf"]),
        "heldout_explained_variance": float(heldout_metrics["global_vaf"]),
        "reconstruction_nrmse": reconstruction_nrmse,
        "heldout_reconstruction_nrmse": reconstruction_nrmse,
        "muscle_coverage": muscle_coverage,
        "active_muscle_coverage": muscle_coverage,
        "local_vaf_coverage_threshold": float(min_local_vaf_coverage),
        "artifact_binding_verified": float(binding_verified),
        # Require every component's full selection contract in addition to
        # content-hash consistency.
        "basis_binding_verified": float(binding_verified and all_regions_eligible),
        "all_regions_eligible": float(all_regions_eligible),
        "regional_component_count": len(descriptors),
        "regional_component_fingerprints": {
            item["region"]: item["component_artifact_fingerprint"] for item in descriptors
        },
        "regional_component_gates": {
            region: {
                "eligible": bool(component_reports[region]["selected_metrics"]["eligible"]),
                "rejection_reasons": component_reports[region]["selected_metrics"]["rejection_reasons"],
                "initialization_similarity": component_reports[region]["selected_metrics"]["initialization_stability"][
                    "mean_similarity"
                ],
                "split_half_similarity": component_reports[region]["selected_metrics"]["split_half_stability"][
                    "mean_similarity"
                ],
                "bootstrap_similarity": component_reports[region]["selected_metrics"]["bootstrap_stability"][
                    "mean_similarity"
                ],
            }
            for region in groups
        },
        "basis_artifact_path": str(artifact.path.resolve()),
        "basis_artifact_fingerprint": artifact.fingerprint,
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "teacher_checkpoint_fingerprint": teacher_checkpoint_fingerprint,
    }
    return {
        "region": "regional_composite",
        "signal_kind": signal.signal_kind,
        "selected_rank": total_rank,
        "selection_reason": "strict_block_diagonal_composition_of_selected_regional_ranks",
        "artifact_path": str(artifact.path.resolve()),
        "artifact_fingerprint": artifact.fingerprint,
        "coefficient_statistics_path": coefficient_stats["path"],
        "coefficient_statistics_fingerprint": coefficient_stats["stats_fingerprint"],
        "artifact_role": "primary_regional_composite",
        "regional_grouping_fingerprint": manifest["regional_grouping_fingerprint"],
        "composite_regions": descriptors,
        "promotion_metrics": _jsonable(promotion_metrics),
    }


def build_hybrid_global_regional_artifact(
    *,
    regional_composite_report: Mapping[str, Any],
    global_report: Mapping[str, Any],
    train_signal: SynergySignal,
    validation_signal: SynergySignal,
    output_path: str | Path,
    teacher_checkpoint_fingerprint: str,
    source_dataset_fingerprint: str,
    split_provenance: Mapping[str, Any],
    train_motion_ids: np.ndarray | None,
    primitive_source_binding: Mapping[str, Any] | None,
    config: SynergyFitConfig,
    dynamic_coverage_reports: Mapping[int | str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Build the primary hybrid decoder from two already-qualified sources."""

    cfg = config.validated()
    signal = train_signal.validated()
    validation = validation_signal.validated()
    _validate_signal_pair(signal, validation)
    source_reports = {
        "regional": regional_composite_report,
        "global": global_report,
    }
    expected_regions = {"regional": "regional_composite", "global": "whole_body"}
    source_artifacts = {}
    source_components: dict[str, dict[str, Any]] = {}
    for label, report in source_reports.items():
        if not isinstance(report, Mapping):
            raise ValueError(f"hybrid {label} source report must be an object")
        supplied_path = Path(str(report.get("artifact_path", ""))).resolve()
        artifact = load_synergy_basis(supplied_path)
        if report.get("artifact_fingerprint") != artifact.fingerprint:
            raise ValueError(f"hybrid {label} source report fingerprint mismatch")
        manifest = artifact.manifest
        if (
            manifest.get("region") != expected_regions[label]
            or manifest.get("signal_kind") != signal.signal_kind
            or manifest.get("teacher_checkpoint_fingerprint") != teacher_checkpoint_fingerprint
            or manifest.get("source_dataset_fingerprint") != source_dataset_fingerprint
            or _json_sha256(manifest.get("split_provenance")) != _json_sha256(split_provenance)
            or _json_sha256(manifest.get("primitive_source_binding"))
            != _json_sha256(primitive_source_binding)
        ):
            raise ValueError(f"hybrid {label} source provenance differs from construction contract")
        if artifact.muscle_names != signal.muscle_names:
            raise ValueError(f"hybrid {label} source muscle schema/order differs from full signal")
        source_artifacts[label] = artifact
        source_components[label] = {
            "region": expected_regions[label],
            "artifact_path": str(supplied_path),
            "artifact_fingerprint": artifact.fingerprint,
        }

    result = build_hybrid_basis(
        source_artifacts["regional"].basis,
        source_artifacts["global"].basis,
        regional_muscle_names=source_artifacts["regional"].muscle_names,
        global_muscle_names=source_artifacts["global"].muscle_names,
        heldout_values=validation.values,
        regional_source_fingerprint=source_artifacts["regional"].fingerprint,
        global_source_fingerprint=source_artifacts["global"].fingerprint,
        config=_hybrid_config_from_fit(cfg),
    )
    result = validate_hybrid_basis_result(
        result,
        regional_basis=source_artifacts["regional"].basis,
        global_basis=source_artifacts["global"].basis,
    )
    total_rank = int(result.basis.shape[1])
    hybrid_region = "hybrid_global_regional"
    candidate_fingerprint = candidate_basis_fingerprint(
        result.basis,
        muscle_names=result.muscle_names,
        signal_kind=signal.signal_kind,
        region=hybrid_region,
    )
    requirement = dynamic_coverage_requirement(
        required=cfg.require_dynamic_coverage,
        max_mean_dynamic_gap=cfg.max_mean_dynamic_gap,
        max_key_phase_dynamic_gap=cfg.max_key_phase_dynamic_gap,
        expected_environment_fingerprint=cfg.expected_environment_fingerprint,
        expected_rollout_manifest_fingerprint=cfg.expected_rollout_manifest_fingerprint,
    )
    validate_dynamic_coverage_rank_inventory(
        dynamic_coverage_reports,
        candidate_ranks=(total_rank,),
        label=f"dynamic coverage inventory for {signal.signal_kind}/{hybrid_region}",
    )
    dynamic_report = dynamic_coverage_report_for_rank(
        dynamic_coverage_reports,
        rank=total_rank,
    )
    validated_dynamic: dict[str, Any] | None = None
    dynamic_validation_error: str | None = None
    if dynamic_report is not None:
        try:
            validated_dynamic = validate_dynamic_coverage_gate(
                dynamic_report,
                region=hybrid_region,
                rank=total_rank,
                candidate_fingerprint=candidate_fingerprint,
                signal_kind=signal.signal_kind,
                max_mean_dynamic_gap=cfg.max_mean_dynamic_gap,
                max_key_phase_dynamic_gap=cfg.max_key_phase_dynamic_gap,
                expected_environment_fingerprint=cfg.expected_environment_fingerprint,
                expected_rollout_manifest_fingerprint=cfg.expected_rollout_manifest_fingerprint,
            )
        except (TypeError, ValueError) as exc:
            dynamic_validation_error = str(exc)

    normalization = {
        "kind": "hybrid_preserves_source_component_units",
        "regional": source_artifacts["regional"].manifest["normalization"],
        "global": source_artifacts["global"].manifest["normalization"],
        "basis_space": "source_signal_units_regional_prefix_plus_original_global_columns",
    }
    transform = None if signal.transform is None else signal.transform.to_manifest()
    if not isinstance(transform, Mapping):
        raise ValueError("hybrid production artifact requires explicit signal transform semantics")
    output = Path(output_path)
    candidate_artifact = None
    candidate_inventory_path = output / "candidate_inventory.json"
    candidate_inventory_fingerprint = None
    if cfg.require_dynamic_coverage:
        candidate_artifact = save_hybrid_basis_artifact(
            output / "candidates" / f"rank_{total_rank:04d}",
            result,
            signal_kind=signal.signal_kind,
            source_dataset_fingerprint=source_dataset_fingerprint,
            teacher_checkpoint_fingerprint=teacher_checkpoint_fingerprint,
            normalization=normalization,
            fit_seed=-1,
            transform=transform,
            split_provenance=split_provenance,
            train_motion_uids=_unique_ints(train_motion_ids),
            primitive_source_binding=primitive_source_binding,
            source_components=source_components,
            artifact_role="dynamic_coverage_rollout_candidate",
            dynamic_coverage_requirement=requirement,
            dynamic_coverage=None,
            candidate_basis_fingerprint=candidate_fingerprint,
        )
        candidate_inventory = {
            "schema_version": "synergy_dynamic_coverage_candidate_inventory_v1",
            "signal_kind": signal.signal_kind,
            "region": hybrid_region,
            "source_dataset_fingerprint": source_dataset_fingerprint,
            "teacher_checkpoint_fingerprint": teacher_checkpoint_fingerprint,
            "dynamic_coverage_requirement": requirement,
            "candidates": [
                {
                    "rank": total_rank,
                    "candidate_basis_fingerprint": candidate_fingerprint,
                    "candidate_artifact_fingerprint": candidate_artifact.fingerprint,
                    "candidate_artifact_path": str(
                        Path("candidates") / f"rank_{total_rank:04d}"
                    ),
                    "offline_eligible": True,
                    "offline_rejection_reasons": [],
                    "dynamic_coverage_report_supplied": dynamic_report is not None,
                    "dynamic_coverage_validation_error": dynamic_validation_error,
                    "dynamic_coverage_passed": (
                        None
                        if validated_dynamic is None
                        else bool(validated_dynamic.get("passed"))
                    ),
                }
            ],
        }
        candidate_inventory_fingerprint = _json_sha256(candidate_inventory)
        candidate_inventory["inventory_fingerprint"] = candidate_inventory_fingerprint
        candidate_inventory_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_inventory_path.write_text(
            json.dumps(
                _jsonable(candidate_inventory),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if (
            dynamic_report is None
            or dynamic_validation_error is not None
            or validated_dynamic is None
            or validated_dynamic.get("passed") is not True
        ):
            return {
                "status": "dynamic_coverage_evidence_required",
                "region": hybrid_region,
                "signal_kind": signal.signal_kind,
                "candidate_inventory_path": str(candidate_inventory_path.resolve()),
                "candidate_inventory_fingerprint": candidate_inventory_fingerprint,
                "unresolved_offline_eligible_ranks": [total_rank],
                "dynamic_coverage_validation_error": dynamic_validation_error,
            }

    artifact = save_hybrid_basis_artifact(
        output,
        result,
        signal_kind=signal.signal_kind,
        source_dataset_fingerprint=source_dataset_fingerprint,
        teacher_checkpoint_fingerprint=teacher_checkpoint_fingerprint,
        normalization=normalization,
        fit_seed=-1,
        transform=transform,
        split_provenance=split_provenance,
        train_motion_uids=_unique_ints(train_motion_ids),
        primitive_source_binding=primitive_source_binding,
        source_components=source_components,
        artifact_role="primary_hybrid_global_regional",
        dynamic_coverage_requirement=requirement,
        dynamic_coverage=validated_dynamic,
        candidate_basis_fingerprint=candidate_fingerprint,
    )
    physical_coefficients, _ = transform_nmf(signal.values, artifact.basis)
    coefficient_stats = save_coefficient_statistics(
        artifact.path / "coefficient_stats.npz",
        physical_coefficients,
        basis_fingerprint=artifact.fingerprint,
    )
    heldout = result.manifest["heldout_evaluation"]
    active_local = [value for value in heldout["heldout_local_vaf"] if value is not None]
    muscle_coverage = float(
        np.mean(
            np.asarray(active_local, dtype=np.float64)
            >= cfg.hybrid_min_heldout_local_vaf_quantile
        )
    )
    promotion_metrics = {
        "schema_version": "forehand_clear_synergy_promotion_metrics_v1",
        "heldout_sample_count": int(validation.values.shape[0]),
        "explained_variance": float(heldout["heldout_global_vaf"]),
        "heldout_explained_variance": float(heldout["heldout_global_vaf"]),
        "muscle_coverage": muscle_coverage,
        "active_muscle_coverage": muscle_coverage,
        "basis_binding_verified": 1.0,
        "artifact_binding_verified": 1.0,
        "basis_artifact_path": str(artifact.path.resolve()),
        "basis_artifact_fingerprint": artifact.fingerprint,
        "source_component_fingerprints": {
            label: source_artifacts[label].fingerprint for label in ("regional", "global")
        },
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "teacher_checkpoint_fingerprint": teacher_checkpoint_fingerprint,
    }
    return {
        "region": hybrid_region,
        "signal_kind": signal.signal_kind,
        "selected_rank": total_rank,
        "selection_reason": "hybrid_static_and_exact_dynamic_gates_passed",
        "artifact_path": str(artifact.path.resolve()),
        "artifact_fingerprint": artifact.fingerprint,
        "coefficient_statistics_path": coefficient_stats["path"],
        "coefficient_statistics_fingerprint": coefficient_stats["stats_fingerprint"],
        "artifact_role": "primary_hybrid_global_regional",
        "selected_metrics": {
            "eligible": True,
            "rejection_reasons": [],
            "hybrid_gate_evidence": heldout,
            "candidate_basis_fingerprint": candidate_fingerprint,
            "dynamic_coverage_required": cfg.require_dynamic_coverage,
            "dynamic_coverage": validated_dynamic,
            "dynamic_coverage_validation_error": dynamic_validation_error,
        },
        "source_components": source_components,
        "promotion_metrics": _jsonable(promotion_metrics),
    }


def _load_explicit_excitation(split: LoadedSynergySplit) -> SynergySignal:
    required = {"teacher_ctrl_physical", "muscle_excitation"}
    missing = sorted(required - set(split.arrays))
    if missing:
        raise ValueError(f"{split.split} excitation fitting is fail-closed and requires raw+unit fields {missing}")
    ctrlrange = validate_unit_muscle_ctrlrange(
        split.muscle_names,
        split.metadata.get("actuator_ctrlrange"),
    )
    expected_source_hash = ordered_schema_hash(
        kind="actuator_ctrlrange",
        payload={
            "actuator_names": list(split.muscle_names),
            "ctrlrange": ctrlrange.tolist(),
        },
    )
    supplied_source_hash = split.metadata.get("ctrlrange_schema_hash")
    if str(supplied_source_hash or "") != expected_source_hash:
        raise ValueError("excitation metadata ctrlrange_schema_hash is missing or mismatched")
    semantics = split.metadata.get("physical_signal_semantics")
    excitation_semantics = semantics.get("muscle_excitation") if isinstance(semantics, dict) else None
    if not isinstance(excitation_semantics, dict):
        raise ValueError("excitation metadata lacks physical_signal_semantics.muscle_excitation")
    if (
        excitation_semantics.get("source") != "teacher_ctrl_physical"
        or excitation_semantics.get("transform") != UNIT_EXCITATION_TRANSFORM
        or excitation_semantics.get("nonnegative") is not True
    ):
        raise ValueError("unsupported or ambiguous persisted excitation transform semantics")
    validate_physical_signal_semantics(semantics)
    capture = split.metadata.get("physical_capture")
    if not isinstance(capture, Mapping) or capture.get("schema_version") != PHYSICAL_CAPTURE_SCHEMA_VERSION:
        raise ValueError(
            "excitation fitting requires physical_capture_spec_v2 metadata; legacy datasets are rejected"
        )
    channel_contract = validate_muscle_channel_contract(
        capture.get("muscle_channel_contract"),
        expected_names=split.muscle_names,
    )
    recomputed = ctrl_to_unit_excitation(
        split.arrays["teacher_ctrl_physical"],
        ctrlrange=ctrlrange,
        actuator_names=split.muscle_names,
        muscle_channel_contract=channel_contract.to_metadata(),
    )
    stored = np.asarray(split.arrays["muscle_excitation"], dtype=np.float64)
    if stored.shape != recomputed.values.shape or not np.allclose(
        stored,
        recomputed.values,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("stored muscle_excitation differs from clip(raw data.ctrl,0,1)")
    # Fit the recomputed evidence, never a trusted-by-name unit array.
    return recomputed


def _validate_signal_pair(train: SynergySignal, val: SynergySignal) -> None:
    if train.signal_kind != val.signal_kind or train.muscle_names != val.muscle_names:
        raise ValueError("train/validation signal kind or ordered muscle schema differs")
    train_transform = None if train.transform is None else train.transform.to_manifest()
    val_transform = None if val.transform is None else val.transform.to_manifest()
    if _json_sha256(train_transform) != _json_sha256(val_transform):
        raise ValueError("train/validation signal transform semantics differ")


def _evaluate_basis(values: np.ndarray, basis: np.ndarray) -> dict[str, Any]:
    _, reconstruction = transform_nmf(values, basis)
    result = reconstruction_metrics(values, reconstruction)
    result["basis_condition_number"] = basis_condition_number(basis)
    result["effective_rank_fraction"] = _basis_effective_rank_fraction(basis)
    finite = np.asarray(result["local_vaf"], dtype=np.float64)
    result["finite_local_vaf_fraction"] = float(np.mean(np.isfinite(finite)))
    return _jsonable(result)


def _basis_effective_rank_fraction(basis: np.ndarray) -> float:
    matrix = np.asarray(basis, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] <= 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("basis must be a finite non-empty matrix")
    return float(np.linalg.matrix_rank(matrix) / matrix.shape[1])


def _primitive_group_validation_metrics(
    values: np.ndarray,
    basis: np.ndarray,
    *,
    task_ids: np.ndarray | None,
    phase_ids: np.ndarray,
    trial_ids: np.ndarray | None,
) -> dict[str, Any]:
    if task_ids is None or trial_ids is None:
        raise ValueError("primitive group validation requires task_ids and trial_ids")
    tasks = np.asarray(task_ids)
    phases = np.asarray(phase_ids)
    trials = np.asarray(trial_ids)
    if tasks.shape != (values.shape[0],) or phases.shape != tasks.shape or trials.shape != tasks.shape:
        raise ValueError("primitive validation task/phase/trial IDs must match held-out rows")
    task_strings = np.asarray([str(value) for value in tasks.tolist()], dtype=object)
    trial_strings = np.asarray([str(value) for value in trials.tolist()], dtype=object)
    per_task: dict[str, Any] = {}
    per_task_phase: dict[str, Any] = {}
    per_trial: dict[str, Any] = {}
    per_task_phase_trial: dict[str, Any] = {}
    global_vafs: list[float] = []
    for task in sorted(set(task_strings.tolist())):
        task_mask = task_strings == task
        if np.count_nonzero(task_mask) < 2:
            raise ValueError(f"primitive validation task {task!r} has fewer than two rows")
        task_metrics = _evaluate_basis(values[task_mask], basis)
        per_task[task] = task_metrics
        global_vafs.append(float(task_metrics["global_vaf"]))
        for phase in np.unique(phases[task_mask]):
            cell_mask = task_mask & (phases == phase)
            if np.count_nonzero(cell_mask) < 2:
                raise ValueError(f"primitive validation task/phase {task!r}/{int(phase)} has fewer than two rows")
            cell_metrics = _evaluate_basis(values[cell_mask], basis)
            per_task_phase[f"{task}::{int(phase)}"] = cell_metrics
            global_vafs.append(float(cell_metrics["global_vaf"]))
            for trial in np.unique(trial_strings[cell_mask]):
                trial_cell_mask = cell_mask & (trial_strings == trial)
                if np.count_nonzero(trial_cell_mask) < 2:
                    raise ValueError(
                        "primitive validation task/phase/trial cell "
                        f"{task!r}/{int(phase)}/{trial!r} has fewer than two rows"
                    )
                trial_cell_metrics = _evaluate_basis(values[trial_cell_mask], basis)
                per_task_phase_trial[f"{task}::{int(phase)}::{trial}"] = trial_cell_metrics
                global_vafs.append(float(trial_cell_metrics["global_vaf"]))
    for trial in sorted(set(trial_strings.tolist())):
        trial_mask = trial_strings == trial
        if np.count_nonzero(trial_mask) < 2:
            raise ValueError(f"primitive validation trial {trial!r} has fewer than two rows")
        trial_metrics = _evaluate_basis(values[trial_mask], basis)
        per_trial[trial] = trial_metrics
        global_vafs.append(float(trial_metrics["global_vaf"]))
    return {
        "per_task": per_task,
        "per_task_phase": per_task_phase,
        "per_trial": per_trial,
        "per_task_phase_trial": per_task_phase_trial,
        "minimum_global_vaf": float(min(global_vafs)),
    }


def _fit_cross_trial_stability(
    values: np.ndarray,
    trial_ids: np.ndarray | None,
    *,
    task_ids: np.ndarray | None = None,
    rank: int,
    seed: int,
    max_iter: int,
    max_trials: int,
) -> dict[str, Any]:
    if trial_ids is None or int(max_trials) <= 1:
        return {"available": False, "reason": "trial IDs unavailable or disabled", "pair_count": 0}
    if task_ids is not None:
        tasks = np.asarray(task_ids)
        ids = np.asarray(trial_ids)
        if tasks.shape != (values.shape[0],) or ids.shape != tasks.shape:
            raise ValueError("primitive task/trial IDs must match training rows")
        per_task: dict[str, Any] = {}
        weighted_similarity = 0.0
        total_pairs = 0
        minimum_similarity = 1.0
        for offset, task in enumerate(sorted({str(value) for value in tasks.tolist()})):
            mask = np.asarray([str(value) == task for value in tasks.tolist()])
            report = _fit_cross_trial_stability(
                values[mask],
                ids[mask],
                rank=rank,
                seed=seed + offset * 10_000,
                max_iter=max_iter,
                max_trials=max_trials,
            )
            per_task[task] = report
            if not report.get("available", False):
                return {
                    "available": False,
                    "reason": f"task {task!r} lacks stable cross-trial evidence",
                    "pair_count": 0,
                    "per_task": per_task,
                }
            pairs = int(report["pair_count"])
            weighted_similarity += float(report["mean_similarity"]) * pairs
            total_pairs += pairs
            minimum_similarity = min(
                minimum_similarity,
                float(report["min_similarity"]),
            )
        return {
            "available": bool(total_pairs > 0),
            "mean_similarity": weighted_similarity / max(1, total_pairs),
            "min_similarity": minimum_similarity,
            "pair_count": total_pairs,
            "per_task": per_task,
        }

    ids = np.asarray([str(value) for value in np.asarray(trial_ids).tolist()], dtype=object)
    if ids.shape != (values.shape[0],):
        raise ValueError("trial IDs must match training rows")
    candidates = [(str(uid), int(np.sum(ids == uid))) for uid in np.unique(ids) if int(np.sum(ids == uid)) >= int(rank)]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    selected = candidates[: int(max_trials)]
    bases: list[np.ndarray] = []
    fitted_trial_ids: list[str] = []
    skipped: dict[str, str] = {}
    for offset, (uid, _) in enumerate(selected):
        try:
            result = fit_nmf(
                values[ids == uid],
                rank=rank,
                seed=seed + offset,
                max_iter=max_iter,
            )
        except (RuntimeError, ValueError) as error:
            skipped[str(uid)] = str(error)
            continue
        bases.append(result.basis)
        fitted_trial_ids.append(uid)
    if len(bases) < 2:
        return {
            "available": False,
            "reason": "fewer than two trials could be fitted at this rank",
            "pair_count": 0,
            "fitted_trial_ids": fitted_trial_ids,
            "skipped": skipped,
        }
    result = cross_trial_stability(bases)
    return {
        "available": True,
        **result,
        "fitted_trial_ids": fitted_trial_ids,
        "skipped": skipped,
    }


def _rank_rejection_reasons(
    *,
    val_global_vaf: float,
    val_local_quantile: float | None,
    initialization: float,
    split_half: float,
    bootstrap: float,
    cross_trial: Mapping[str, Any],
    primitive_group_min_vaf: float | None,
    basis_condition_number_value: float,
    effective_rank_fraction: float,
    config: SynergyFitConfig,
) -> list[str]:
    reasons: list[str] = []
    if val_global_vaf < config.min_val_global_vaf:
        reasons.append("heldout_global_vaf_below_threshold")
    if val_local_quantile is None or val_local_quantile < config.min_val_local_vaf_quantile:
        reasons.append("heldout_local_vaf_quantile_below_threshold_or_undefined")
    if primitive_group_min_vaf is not None and primitive_group_min_vaf < config.min_val_local_vaf_quantile:
        reasons.append("primitive_task_phase_global_vaf_below_threshold")
    if initialization < config.min_initialization_similarity:
        reasons.append("initialization_stability_below_threshold")
    if split_half < config.min_split_half_similarity:
        reasons.append("split_half_stability_below_threshold")
    if bootstrap < config.min_bootstrap_similarity:
        reasons.append("bootstrap_stability_below_threshold")
    if cross_trial.get("available"):
        cross_trial_failed = float(cross_trial["mean_similarity"]) < config.min_cross_trial_similarity
        per_task = cross_trial.get("per_task")
        if isinstance(per_task, Mapping):
            cross_trial_failed = cross_trial_failed or any(
                float(report["mean_similarity"]) < config.min_cross_trial_similarity
                or float(report["min_similarity"]) < config.min_cross_trial_similarity
                for report in per_task.values()
            )
        if cross_trial_failed:
            reasons.append("cross_trial_stability_below_threshold")
    if not cross_trial.get("available"):
        reasons.append("cross_trial_stability_unavailable")
    if (
        not np.isfinite(basis_condition_number_value)
        or basis_condition_number_value > config.max_basis_condition_number
    ):
        reasons.append("basis_condition_number_above_threshold_or_nonfinite")
    if (
        not np.isfinite(effective_rank_fraction)
        or effective_rank_fraction < config.min_effective_rank_fraction
    ):
        reasons.append("effective_rank_fraction_below_threshold_or_nonfinite")
    return reasons


def _load_npz_fields(files: Sequence[Path]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    loaded: dict[str, list[np.ndarray]] = {}
    embedded: dict[str, Any] = {}
    common_fields: set[str] | None = None
    for path in files:
        with np.load(path, allow_pickle=False) as shard:
            fields = set(shard.files)
            common_fields = fields if common_fields is None else common_fields & fields
            for name in ("actuator_names", "actuator_ctrlrange", "metadata_json"):
                if name in shard:
                    value = np.asarray(shard[name])
                    parsed = value.tolist()
                    if name == "metadata_json":
                        parsed = loads_json_strict(str(parsed))
                    previous = embedded.get(name)
                    if previous is not None and _json_sha256(previous) != _json_sha256(parsed):
                        raise ValueError(f"embedded {name} differs across shards")
                    embedded[name] = parsed
            for field in _SAMPLED_FIELDS:
                if field in shard:
                    loaded.setdefault(field, []).append(np.asarray(shard[field]))
    common_fields = common_fields or set()
    partial = sorted(field for field in loaded if field not in common_fields)
    if partial:
        raise ValueError(f"synergy fields are present in only a subset of shards: {partial}")
    arrays = {field: np.concatenate(parts, axis=0) for field, parts in loaded.items()}
    if not arrays:
        raise ValueError("synergy source contains none of the required physical/event fields")
    sample_counts = {int(array.shape[0]) for array in arrays.values()}
    if len(sample_counts) != 1:
        raise ValueError(f"synergy source fields have inconsistent sample counts: {sample_counts}")
    return arrays, embedded


def _merge_embedded_metadata(metadata: dict[str, Any], embedded: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    nested = embedded.get("metadata_json")
    if isinstance(nested, dict):
        for key, value in nested.items():
            if key in result and _json_sha256(result[key]) != _json_sha256(value):
                raise ValueError(f"metadata key {key!r} conflicts with embedded metadata_json")
            result[key] = value
    mapping = {"actuator_names": "actuator_names", "actuator_ctrlrange": "actuator_ctrlrange"}
    for embedded_key, metadata_key in mapping.items():
        if embedded_key not in embedded:
            continue
        value = embedded[embedded_key]
        if metadata_key in result and _json_sha256(result[metadata_key]) != _json_sha256(value):
            raise ValueError(f"metadata {metadata_key} conflicts with embedded NPZ value")
        result[metadata_key] = value
    return result


def _resolve_teacher_fingerprint(
    explicit: str | None,
    train_metadata: Mapping[str, Any],
    val_metadata: Mapping[str, Any],
) -> str:
    candidates = {
        _require_sha256(value, field="teacher_checkpoint_fingerprint")
        for value in (
            explicit,
            train_metadata.get("teacher_checkpoint_fingerprint"),
            val_metadata.get("teacher_checkpoint_fingerprint"),
        )
        if value is not None and str(value).strip()
    }
    if not candidates:
        raise ValueError("teacher_checkpoint_fingerprint is required; a checkpoint path is not a fingerprint")
    if len(candidates) != 1:
        raise ValueError(f"teacher checkpoint fingerprints disagree: {sorted(candidates)}")
    resolved = candidates.pop()
    for split, metadata in (("train", train_metadata), ("validation", val_metadata)):
        content = metadata.get("teacher_checkpoint_content")
        if not isinstance(content, Mapping):
            raise ValueError(f"{split} synergy metadata lacks teacher_checkpoint_content audit record")
        if content.get("schema_version") != "checkpoint_content_fingerprint_v1":
            raise ValueError(f"{split} teacher checkpoint content schema is unsupported")
        content_sha = _require_sha256(
            content.get("sha256"),
            field=f"{split} teacher_checkpoint_content.sha256",
        )
        if content_sha != resolved:
            raise ValueError(f"{split} teacher checkpoint content record differs from compact fingerprint")
        files = content.get("files")
        if (
            not isinstance(files, list)
            or not files
            or int(content.get("num_files", -1)) != len(files)
            or int(content.get("num_bytes", -1)) < 0
        ):
            raise ValueError(f"{split} teacher checkpoint content inventory is incomplete")
    return resolved


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{field} must be a lowercase 64-hex SHA-256")
    return digest


def _source_fingerprint(files: Sequence[Path], metadata: Mapping[str, Any]) -> str:
    return _json_sha256(
        {
            "schema_version": "synergy_source_content_v1",
            "files": [{"name": path.name, "sha256": _file_sha256(path)} for path in files],
            "metadata": _jsonable(metadata),
        }
    )


def _canonical_signal_kind(value: str) -> str:
    key = str(value).strip()
    if key not in _SIGNAL_ALIASES:
        raise ValueError(f"unsupported signal kind {value!r}; use excitation and/or activation")
    return _SIGNAL_ALIASES[key]


def _validate_dynamic_coverage_inventory(
    reports: Mapping[str, Mapping[str, Mapping[int | str, Mapping[str, Any]]]] | None,
    *,
    signal_kinds: Sequence[str],
    region_candidate_ranks: Mapping[str, Sequence[int]],
    allow_hybrid: bool = False,
) -> None:
    """Reject stale or silently ignored dynamic-evidence inventory entries."""

    if reports is None:
        return
    if not isinstance(reports, Mapping):
        raise TypeError("dynamic_coverage_reports must be keyed by signal kind, region, then rank")
    expected_kinds = set(signal_kinds)
    if set(reports) - expected_kinds:
        raise ValueError(
            "dynamic coverage inventory contains unrequested signal kinds: "
            f"{sorted(set(reports) - expected_kinds)}"
        )
    expected_regions = set(region_candidate_ranks)
    for signal_kind, region_reports in reports.items():
        if not isinstance(region_reports, Mapping):
            raise TypeError(f"dynamic coverage inventory {signal_kind!r} must be region-keyed")
        allowed_regions = expected_regions | ({"hybrid_global_regional"} if allow_hybrid else set())
        unknown_regions = set(region_reports) - allowed_regions
        if unknown_regions:
            raise ValueError(
                f"dynamic coverage inventory {signal_kind!r} contains unknown regions: {sorted(unknown_regions)}"
            )
        for region, rank_reports in region_reports.items():
            if region == "hybrid_global_regional":
                if not isinstance(rank_reports, Mapping) or not rank_reports:
                    raise TypeError(
                        f"dynamic coverage inventory {signal_kind!r}/{region!r} must be non-empty and rank-keyed"
                    )
                observed_ranks: list[int] = []
                for raw_rank in rank_reports:
                    if isinstance(raw_rank, bool):
                        raise ValueError("hybrid dynamic coverage rank keys cannot be boolean")
                    try:
                        rank = int(raw_rank)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("hybrid dynamic coverage rank keys must be positive integers") from exc
                    if rank <= 0 or str(rank) != str(raw_rank):
                        raise ValueError("hybrid dynamic coverage rank keys must be canonical positive integers")
                    observed_ranks.append(rank)
                validate_dynamic_coverage_rank_inventory(
                    rank_reports,
                    candidate_ranks=observed_ranks,
                    label=f"dynamic coverage inventory {signal_kind!r}/{region!r}",
                )
                continue
            validate_dynamic_coverage_rank_inventory(
                rank_reports,
                candidate_ranks=region_candidate_ranks[region],
                label=f"dynamic coverage inventory {signal_kind!r}/{region!r}",
            )


def _dynamic_coverage_reports_for_region(
    reports: Mapping[str, Mapping[str, Mapping[int | str, Mapping[str, Any]]]] | None,
    *,
    signal_kind: str,
    region: str,
) -> Mapping[int | str, Mapping[str, Any]] | None:
    if reports is None:
        return None
    signal_reports = reports.get(signal_kind)
    if signal_reports is None:
        return None
    return signal_reports.get(region)


def _finite_quantile(values: Sequence[float | None], quantile: float) -> float | None:
    finite = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(float(value))],
        dtype=np.float64,
    )
    return None if finite.size == 0 else float(np.quantile(finite, float(quantile)))


def _unique_ints(values: np.ndarray | None) -> list[int]:
    if values is None:
        return []
    return [int(value) for value in np.unique(np.asarray(values, dtype=np.int64)).tolist()]


def _safe_slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not result:
        result = "region_" + hashlib.sha256(str(value).encode()).hexdigest()[:12]
    return result


def _invalidate_hybrid_primary_artifact(path: str | Path) -> None:
    root = Path(path)
    for filename in ("basis.npy", "manifest.json", "coefficient_stats.npz"):
        _invalidate_output_file(root / filename)


def _invalidate_output_file(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()
    elif path.exists():
        raise ValueError(f"expected output file path is occupied by a non-file: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _parse_phase_weights(path: str | None) -> Mapping[int, float] | None:
    if path is None:
        return None
    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise ValueError("phase weights JSON must be an object mapping phase id to weight")
    return {int(key): float(value) for key, value in payload.items()}


def _parse_region_ranks(path: str | None) -> Mapping[str, tuple[int, ...]] | None:
    if path is None:
        return None
    payload = load_json_strict(path)
    if not isinstance(payload, Mapping):
        raise ValueError("region ranks JSON must be an object mapping region to rank list")
    return canonical_region_candidate_ranks(payload)


def _parse_dynamic_coverage_reports(
    path: str | None,
) -> Mapping[str, Mapping[str, Mapping[int | str, Mapping[str, Any]]]] | None:
    if path is None:
        return None
    payload = load_json_strict(path)
    if not isinstance(payload, Mapping):
        raise ValueError("dynamic coverage reports JSON must contain an object")
    return payload


def synergy_preprocessing_fingerprint(config: SynergyFitConfig) -> str:
    """Identity of preprocessing choices that change the fitted matrix."""

    cfg = config.validated()
    return _json_sha256(
        {
            "schema_version": "synergy_preprocessing_contract_v1",
            "normalization": cfg.normalization,
            "near_zero_threshold": cfg.near_zero_threshold,
        }
    )


def primitive_task_phase_balanced_weights(
    task_ids: np.ndarray | None,
    phase_ids: np.ndarray,
    *,
    trial_ids: np.ndarray | None,
    quality_weights: np.ndarray | None,
    phase_weights: Mapping[int, float] | None,
) -> np.ndarray:
    """Balance primitive task-by-phase cells before applying rollout QC weights."""

    if task_ids is None or trial_ids is None or quality_weights is None:
        raise ValueError("primitive balancing requires task ids, trial ids, and quality weights")
    tasks = np.asarray(task_ids)
    trials = np.asarray(trial_ids)
    phases = np.asarray(phase_ids)
    quality = np.asarray(quality_weights, dtype=np.float64)
    if tasks.ndim != 1 or trials.shape != tasks.shape or phases.shape != tasks.shape or quality.shape != tasks.shape:
        raise ValueError("primitive task, trial, phase, and quality arrays must share shape [samples]")
    if tasks.size == 0 or not np.issubdtype(phases.dtype, np.integer):
        raise ValueError("primitive task/phase balancing requires non-empty integer phases")
    if not np.all(np.isfinite(quality)) or np.any(quality <= 0.0):
        raise ValueError("primitive quality weights must be finite and positive")
    configured = phase_weights or DEFAULT_PHASE_WEIGHTS
    result = np.zeros(tasks.shape[0], dtype=np.float64)
    string_tasks = np.asarray([str(value) for value in tasks.tolist()], dtype=object)
    string_trials = np.asarray([str(value) for value in trials.tolist()], dtype=object)
    unique_tasks = np.unique(string_tasks)
    task_total = 1.0 / float(unique_tasks.size)
    for task in unique_tasks:
        task_mask = string_tasks == task
        task_phases = np.unique(phases[task_mask])
        missing = [int(value) for value in task_phases if int(value) not in configured]
        if missing:
            raise ValueError(f"phase weights missing primitive task phases {missing}")
        phase_denominator = sum(float(configured[int(value)]) for value in task_phases)
        if phase_denominator <= 0.0:
            raise ValueError("primitive task has no positive configured phase weight")
        for phase in task_phases:
            cell_mask = task_mask & (phases == phase)
            phase_total = task_total * float(configured[int(phase)]) / phase_denominator
            cell_trials = np.unique(string_trials[cell_mask])
            trial_quality_means = np.asarray(
                [float(np.mean(quality[cell_mask & (string_trials == trial)])) for trial in cell_trials],
                dtype=np.float64,
            )
            trial_totals = phase_total * trial_quality_means / float(np.sum(trial_quality_means))
            for trial, trial_total in zip(cell_trials, trial_totals, strict=True):
                trial_mask = cell_mask & (string_trials == trial)
                trial_quality = quality[trial_mask]
                result[trial_mask] = trial_total * trial_quality / float(np.sum(trial_quality))
    mean = float(np.mean(result))
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError("primitive task/phase balancing produced invalid sample weights")
    return result / mean


def synergy_phase_weight_fingerprint(config: SynergyFitConfig) -> str:
    """Identity of the phase-balancing rule applied during primitive NMF."""

    cfg = config.validated()
    return _json_sha256(
        {
            "schema_version": "synergy_phase_weight_contract_v2",
            "phase_weights": {str(key): float(value) for key, value in sorted((cfg.phase_weights or {}).items())},
            "sample_weight_application": "multiply_rows_by_sqrt_weight",
            "fit_scope": "train_only",
            "primitive_cell_balance": "equal_task_then_phase_weight",
            "trial_quality_application": "allocate_phase_total_proportional_to_trial_mean_quality",
            "frame_quality_application": "normalize_within_trial_after_trial_total_allocation",
        }
    )


def _load_primitive_source_binding(
    source_manifest: str | Path | None,
    *,
    source_dataset_fingerprint: str,
    train_motion_uids: set[int],
    validation_motion_uids: set[int],
    actuator_names: Sequence[str],
    train_metadata: Mapping[str, Any],
    validation_metadata: Mapping[str, Any],
    train_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
    fit_config: SynergyFitConfig,
) -> dict[str, Any] | None:
    if source_manifest is None:
        return None
    source = load_primitive_source_manifest(source_manifest)
    manifest = source.manifest
    if manifest["source_dataset_fingerprint"] != source_dataset_fingerprint:
        raise ValueError(
            "primitive source manifest dataset fingerprint differs from the loaded train/validation shards"
        )
    actual_train = {int(value) for value in train_motion_uids}
    actual_validation = {int(value) for value in validation_motion_uids}
    if set(manifest["train_motion_uids"]) != actual_train:
        raise ValueError("primitive source manifest train motion_uids differ from loaded shards")
    if set(manifest["validation_motion_uids"]) != actual_validation:
        raise ValueError("primitive source manifest validation motion_uids differ from loaded shards")
    if manifest["actuator_schema_hash"] != actuator_schema_hash(actuator_names):
        raise ValueError("primitive source manifest actuator schema differs from loaded shards")
    for split, metadata in (
        ("train", train_metadata),
        ("validation", validation_metadata),
    ):
        if metadata.get("ctrlrange_schema_hash") != manifest["control_range_hash"]:
            raise ValueError(f"primitive source manifest control range differs from {split} shard metadata")
        model_hash = metadata.get("model_hash", metadata.get("model_fingerprint"))
        if model_hash != manifest["model_hash"]:
            raise ValueError(f"primitive source manifest model hash differs from {split} shard metadata")
        if metadata.get("source_checkpoint_fingerprints") != manifest["source_checkpoint_fingerprints"]:
            raise ValueError(f"primitive source checkpoint inventory differs from {split} shard metadata")
        if metadata.get("source_checkpoint_contents") != manifest["source_checkpoint_contents"]:
            raise ValueError(f"primitive source checkpoint content audit differs from {split} shard metadata")
        if metadata.get("primitive_required_phase_ids") != manifest["primitive_required_phase_ids"]:
            raise ValueError(f"primitive required-phase inventory differs from {split} shard metadata")
        if metadata.get("primitive_phase_schema_fingerprints") != manifest["primitive_phase_schema_fingerprints"]:
            raise ValueError(f"primitive phase-schema inventory differs from {split} shard metadata")
    train_ctrlrange = np.asarray(train_metadata.get("actuator_ctrlrange"), dtype=np.float64)
    validation_ctrlrange = np.asarray(
        validation_metadata.get("actuator_ctrlrange"),
        dtype=np.float64,
    )
    expected_shape = (len(actuator_names), 2)
    if (
        train_ctrlrange.shape != expected_shape
        or validation_ctrlrange.shape != expected_shape
        or not np.array_equal(train_ctrlrange, validation_ctrlrange)
    ):
        raise ValueError("primitive train/validation actuator control ranges differ")
    if manifest["transform_ctrlrange_schema_hash"] != ctrlrange_schema_hash(
        actuator_names,
        train_ctrlrange,
    ):
        raise ValueError("primitive source transform ctrlrange hash differs from loaded shards")
    _validate_primitive_sample_inventory(
        train_arrays,
        validation_arrays,
        manifest=manifest,
    )
    if manifest["preprocessing_fingerprint"] != synergy_preprocessing_fingerprint(fit_config):
        raise ValueError("primitive source preprocessing fingerprint differs from fit config")
    if manifest["phase_weight_fingerprint"] != synergy_phase_weight_fingerprint(fit_config):
        raise ValueError("primitive source phase-weight fingerprint differs from fit config")
    if tuple(manifest["NMF_seeds"]) != tuple(fit_config.seeds):
        raise ValueError("primitive source NMF_seeds differ from configured fit seeds")
    return {
        "schema_version": manifest["schema_version"],
        "manifest_fingerprint": source.fingerprint,
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "primitive_only": True,
        "contains_target_skill_rollouts": False,
        "target_skill_id": manifest["target_skill_id"],
        "excluded_target_motions": list(manifest["excluded_target_motions"]),
        "primitive_task_ids": list(manifest["primitive_task_ids"]),
        "model_hash": manifest["model_hash"],
        "transform_ctrlrange_schema_hash": manifest["transform_ctrlrange_schema_hash"],
    }


def _validate_primitive_sample_inventory(
    train_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
    *,
    manifest: Mapping[str, Any],
) -> None:
    required = {
        "phase_id",
        "motion_uid",
        "task_id",
        "trial_id",
        "source_kind",
        "success",
        "quality_weight",
    }
    expected_tasks = set(manifest["primitive_task_ids"])
    split_trials: dict[str, set[str]] = {}
    all_trial_to_task: dict[str, str] = {}
    all_trial_to_motion: dict[str, int] = {}
    for split, arrays in (("train", train_arrays), ("validation", validation_arrays)):
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(f"primitive {split} shards lack source/QC fields: {missing}")
        sample_count = int(np.asarray(arrays["phase_id"]).shape[0])
        task = np.asarray(arrays["task_id"])
        trial = np.asarray(arrays["trial_id"])
        source_kind = np.asarray(arrays["source_kind"])
        success = np.asarray(arrays["success"])
        quality = np.asarray(arrays["quality_weight"], dtype=np.float64)
        motion_uid = np.asarray(arrays["motion_uid"])
        phase_id = np.asarray(arrays["phase_id"])
        for label, values in (
            ("task_id", task),
            ("trial_id", trial),
            ("source_kind", source_kind),
            ("success", success),
            ("quality_weight", quality),
            ("motion_uid", motion_uid),
        ):
            if values.shape != (sample_count,):
                raise ValueError(f"primitive {split} {label} must have shape [{sample_count}]")
        for label, values in (
            ("task_id", task),
            ("trial_id", trial),
            ("source_kind", source_kind),
        ):
            if values.dtype.kind not in {"U", "S"}:
                raise ValueError(f"primitive {split} {label} must use a string dtype")
        if np.issubdtype(motion_uid.dtype, np.bool_) or not np.issubdtype(
            motion_uid.dtype,
            np.integer,
        ):
            raise ValueError(f"primitive {split} motion_uid must use an integer dtype")
        if np.issubdtype(phase_id.dtype, np.bool_) or not np.issubdtype(
            phase_id.dtype,
            np.integer,
        ):
            raise ValueError(f"primitive {split} phase_id must use an integer dtype")
        if any(str(value) != "primitive" for value in source_kind.tolist()):
            raise ValueError(f"primitive {split} shards contain non-primitive source_kind")
        if not np.all(success.astype(bool)) or np.any(np.asarray(success, dtype=np.float64) != 1.0):
            raise ValueError(f"primitive {split} shards contain unsuccessful samples")
        if not np.all(np.isfinite(quality)) or np.any(quality <= 0.0):
            raise ValueError(f"primitive {split} quality_weight must be finite and positive")
        task_strings = np.asarray([str(value) for value in task.tolist()], dtype=object)
        trial_strings = np.asarray([str(value) for value in trial.tolist()], dtype=object)
        observed_tasks = set(task_strings.tolist())
        if observed_tasks != expected_tasks:
            raise ValueError(f"primitive {split} task inventory must cover every declared primitive")
        split_trials[split] = set(trial_strings.tolist())
        minimum_trials = 2 if split == "train" else 1
        for task_id in sorted(expected_tasks):
            observed_phases = {int(value) for value in phase_id[task_strings == task_id].tolist()}
            required_phases = set(manifest["primitive_required_phase_ids"][task_id])
            missing_phases = sorted(required_phases - observed_phases)
            if missing_phases:
                raise ValueError(f"primitive {split} task {task_id!r} is missing required phase_ids: {missing_phases}")
            task_trials = set(trial_strings[task_strings == task_id].tolist())
            if len(task_trials) < minimum_trials:
                raise ValueError(
                    f"primitive {split} task {task_id!r} requires at least {minimum_trials} distinct trial(s)"
                )
            for trial_id in task_trials:
                previous = all_trial_to_task.setdefault(trial_id, task_id)
                if previous != task_id:
                    raise ValueError("primitive trial_id cannot belong to multiple tasks")
                trial_motion_uids = {int(value) for value in motion_uid[trial_strings == trial_id].tolist()}
                if len(trial_motion_uids) != 1:
                    raise ValueError("primitive trial_id must bind exactly one motion_uid")
                motion = next(iter(trial_motion_uids))
                previous_motion = all_trial_to_motion.setdefault(trial_id, motion)
                if previous_motion != motion:
                    raise ValueError("primitive trial_id motion binding changed across shards")
    if split_trials["train"] & split_trials["validation"]:
        raise ValueError("primitive train/validation trial_id inventories overlap")
    trial_ids = split_trials["train"] | split_trials["validation"]
    if trial_ids != set(manifest["primitive_trial_ids"]):
        raise ValueError("primitive trial inventory differs from sampled shard content")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Train dataset directory or one NPZ shard")
    parser.add_argument("--val", required=True, help="Validation dataset directory or one NPZ shard")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-checkpoint-fingerprint", default=None)
    parser.add_argument("--signals", nargs="+", default=["excitation", "activation"])
    parser.add_argument("--mode", choices=["global", "regional", "both"], default="both")
    parser.add_argument("--grouping-json", default=None)
    parser.add_argument(
        "--anatomical-taxonomy-json",
        default=None,
        help=(
            "optional anatomy taxonomy manifest; binds the regional grouping's ordered "
            "muscle schema to the compiled model's ordered actuators instead of only to "
            "the training dataset order"
        ),
    )
    parser.add_argument(
        "--primitive-source-manifest",
        default=None,
        help="strict source_manifest.json required for a primitive-only early-control basis",
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument(
        "--region-ranks-json",
        default=None,
        help="path to an optional JSON object mapping each region to its candidate rank list",
    )
    parser.add_argument(
        "--total-rank-budget",
        type=int,
        default=None,
        help="fail if the selected regional composite rank exceeds this budget",
    )
    parser.add_argument(
        "--require-dynamic-coverage",
        action="store_true",
        help="require separately sealed environment-rollout coverage for the selected rank",
    )
    parser.add_argument(
        "--dynamic-coverage-reports-json",
        default=None,
        help="path to a strict signal-kind/region/rank dynamic coverage evidence inventory",
    )
    parser.add_argument("--max-mean-dynamic-gap", type=float, default=0.15)
    parser.add_argument("--max-key-phase-dynamic-gap", type=float, default=0.25)
    parser.add_argument(
        "--expected-environment-fingerprint",
        default=None,
        help="lowercase SHA-256 of the exact rollout environment contract",
    )
    parser.add_argument(
        "--expected-rollout-manifest-fingerprint",
        default=None,
        help="lowercase SHA-256 of the exact rollout request/trajectory manifest",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--normalization", choices=["channel_max", "channel_l2", "none"], default="channel_max")
    parser.add_argument("--near-zero-threshold", type=float, default=1e-8)
    parser.add_argument("--phase-weights-json", default=None)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--split-half-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=10)
    parser.add_argument("--cross-trial-max-trials", type=int, default=12)
    parser.add_argument("--min-val-global-vaf", type=float, default=0.90)
    parser.add_argument("--min-val-local-vaf-quantile", type=float, default=0.70)
    parser.add_argument("--local-vaf-quantile", type=float, default=0.10)
    parser.add_argument("--min-initialization-similarity", type=float, default=0.80)
    parser.add_argument("--min-split-half-similarity", type=float, default=0.80)
    parser.add_argument("--min-bootstrap-similarity", type=float, default=0.80)
    parser.add_argument("--min-cross-trial-similarity", type=float, default=0.75)
    parser.add_argument("--max-basis-condition-number", type=float, default=1.0e6)
    parser.add_argument("--min-effective-rank-fraction", type=float, default=1.0)
    parser.add_argument("--hybrid-novelty-residual-ratio", type=float, default=0.15)
    parser.add_argument("--hybrid-duplicate-cosine-similarity", type=float, default=0.95)
    parser.add_argument("--hybrid-min-heldout-global-vaf-marginal-gain", type=float, default=1e-6)
    parser.add_argument("--hybrid-max-total-rank", type=int, default=64)
    parser.add_argument("--hybrid-min-heldout-global-vaf", type=float, default=0.90)
    parser.add_argument("--hybrid-local-vaf-quantile", type=float, default=0.10)
    parser.add_argument("--hybrid-min-heldout-local-vaf-quantile", type=float, default=0.70)
    parser.add_argument("--hybrid-max-basis-condition-number", type=float, default=100.0)
    parser.add_argument("--hybrid-min-effective-rank-fraction", type=float, default=0.80)
    parser.add_argument("--hybrid-effective-rank-relative-tolerance", type=float, default=1e-8)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = SynergyFitConfig(
        ranks=tuple(args.ranks),
        region_ranks=_parse_region_ranks(args.region_ranks_json),
        total_rank_budget=args.total_rank_budget,
        require_dynamic_coverage=args.require_dynamic_coverage,
        max_mean_dynamic_gap=args.max_mean_dynamic_gap,
        max_key_phase_dynamic_gap=args.max_key_phase_dynamic_gap,
        expected_environment_fingerprint=args.expected_environment_fingerprint,
        expected_rollout_manifest_fingerprint=(
            args.expected_rollout_manifest_fingerprint
        ),
        seeds=tuple(args.seeds),
        normalization=args.normalization,
        near_zero_threshold=args.near_zero_threshold,
        phase_weights=_parse_phase_weights(args.phase_weights_json),
        max_iter=args.max_iter,
        tol=args.tol,
        split_half_repeats=args.split_half_repeats,
        bootstrap_repeats=args.bootstrap_repeats,
        cross_trial_max_trials=args.cross_trial_max_trials,
        min_val_global_vaf=args.min_val_global_vaf,
        min_val_local_vaf_quantile=args.min_val_local_vaf_quantile,
        local_vaf_quantile=args.local_vaf_quantile,
        min_initialization_similarity=args.min_initialization_similarity,
        min_split_half_similarity=args.min_split_half_similarity,
        min_bootstrap_similarity=args.min_bootstrap_similarity,
        min_cross_trial_similarity=args.min_cross_trial_similarity,
        max_basis_condition_number=args.max_basis_condition_number,
        min_effective_rank_fraction=args.min_effective_rank_fraction,
        hybrid_novelty_residual_ratio=args.hybrid_novelty_residual_ratio,
        hybrid_duplicate_cosine_similarity=args.hybrid_duplicate_cosine_similarity,
        hybrid_min_heldout_global_vaf_marginal_gain=(
            args.hybrid_min_heldout_global_vaf_marginal_gain
        ),
        hybrid_max_total_rank=args.hybrid_max_total_rank,
        hybrid_min_heldout_global_vaf=args.hybrid_min_heldout_global_vaf,
        hybrid_local_vaf_quantile=args.hybrid_local_vaf_quantile,
        hybrid_min_heldout_local_vaf_quantile=(
            args.hybrid_min_heldout_local_vaf_quantile
        ),
        hybrid_max_basis_condition_number=args.hybrid_max_basis_condition_number,
        hybrid_min_effective_rank_fraction=args.hybrid_min_effective_rank_fraction,
        hybrid_effective_rank_relative_tolerance=(
            args.hybrid_effective_rank_relative_tolerance
        ),
    )
    report = fit_synergy_dataset(
        args.train,
        args.val,
        output_dir=args.output_dir,
        teacher_checkpoint_fingerprint=args.teacher_checkpoint_fingerprint,
        signal_kinds=args.signals,
        mode=args.mode,
        grouping_json=args.grouping_json,
        anatomical_taxonomy_json=args.anatomical_taxonomy_json,
        primitive_source_manifest=args.primitive_source_manifest,
        config=config,
        dynamic_coverage_reports=_parse_dynamic_coverage_reports(
            args.dynamic_coverage_reports_json
        ),
    )
    print(
        json.dumps(
            {
                "report": str((Path(args.output_dir) / "fit_report.json").resolve()),
                "source_dataset_fingerprint": report["source_dataset_fingerprint"],
                "artifacts": [
                    {
                        "signal_kind": item["signal_kind"],
                        "region": item["region"],
                        "rank": item["selected_rank"],
                        "fingerprint": item["artifact_fingerprint"],
                    }
                    for item in report["artifacts"]
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
