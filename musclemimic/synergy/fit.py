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
    UNIT_EXCITATION_TRANSFORM,
    UNIT_INTERVAL_ROUNDOFF_POLICY,
    validate_activation_valid_mask,
    validate_physical_signal_semantics,
    validate_unit_muscle_activation,
)
from musclemimic.synergy.action_interface import save_coefficient_statistics
from musclemimic.synergy.basis_artifact import load_synergy_basis, save_synergy_basis
from musclemimic.synergy.collect import ctrl_to_unit_excitation
from musclemimic.synergy.grouping import global_group, load_grouping_json
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


class BasisNotEligibleForEarlyControl(ValueError):  # noqa: N818
    """Raised when no primitive basis rank passes every early-control gate."""


@dataclass(frozen=True)
class SynergyFitConfig:
    ranks: tuple[int, ...] = tuple(range(1, 11))
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

    def validated(self) -> SynergyFitConfig:
        ranks = tuple(sorted({int(value) for value in self.ranks}))
        seeds = tuple(int(value) for value in self.seeds)
        if not ranks or min(ranks) <= 0:
            raise ValueError("ranks must contain positive integers")
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
        weights = self.phase_weights or DEFAULT_PHASE_WEIGHTS
        phase_balanced_weights(np.asarray(sorted(weights), dtype=np.int32), weights=weights)
        return SynergyFitConfig(
            **{
                **asdict(self),
                "ranks": ranks,
                "seeds": seeds,
                "phase_weights": {int(key): float(value) for key, value in weights.items()},
            }
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
        if not isinstance(capture, Mapping) or capture.get("schema_version") != "physical_capture_spec_v1":
            raise ValueError(f"{self.split} activation fitting requires physical_capture_spec_v1 metadata")
        capture_names = tuple(str(name) for name in capture.get("actuator_names", ()))
        if capture_names != self.muscle_names:
            raise ValueError(f"{self.split} activation capture actuator order differs from dataset metadata")
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
        )
    elif transform is not None:
        transform = SignalTransform(
            kind=transform.kind,
            raw_signal_kind=transform.raw_signal_kind,
            formula=transform.formula,
            actuator_names=names,
            roundoff_policy=transform.roundoff_policy,
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

    max_rank = min(weighted_train.shape)
    ranks = tuple(rank for rank in cfg.ranks if rank <= max_rank)
    rejected_ranks = tuple(rank for rank in cfg.ranks if rank > max_rank)
    if not ranks:
        raise ValueError(f"no candidate rank fits preprocessed matrix {weighted_train.shape}; requested {cfg.ranks}")

    rank_reports: dict[int, dict[str, Any]] = {}
    best_results: dict[int, NMFResult] = {}
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
        rejection_reasons = _rank_rejection_reasons(
            val_global_vaf=float(eligibility_metrics["global_vaf"]),
            val_local_quantile=local_quantile,
            initialization=float(init_report["mean_similarity"]),
            split_half=float(split_report["mean_similarity"]),
            bootstrap=float(bootstrap_report["mean_similarity"]),
            cross_trial=cross_trial_report,
            primitive_group_min_vaf=(
                None if primitive_group_validation is None else float(primitive_group_validation["minimum_global_vaf"])
            ),
            config=cfg,
        )
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
            "eligible": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
        }

    eligible = [rank for rank in ranks if rank_reports[rank]["eligible"]]
    if eligible:
        selected_rank = min(eligible)
        selection_reason = "smallest_rank_meeting_all_vaf_and_stability_gates"
    else:
        if primitive_source_binding is not None:
            raise BasisNotEligibleForEarlyControl(
                f"primitive region {region!r} has no rank passing every VAF/stability gate"
            )
        selected_rank = max(
            ranks,
            key=lambda rank: (
                float(rank_reports[rank]["validation"]["global_vaf"]),
                -rank,
            ),
        )
        selection_reason = "fallback_best_heldout_global_vaf_no_rank_met_all_gates"
    selected = best_results[selected_rank]

    # NMF is fitted in normalized coordinates.  Undo only the channel scaling
    # and restore removed near-zero rows so the saved W consumes the complete,
    # ordered physical signal schema expected by a decoder.
    physical_kept_basis = preprocess.scales[:, None] * selected.basis
    physical_basis = np.zeros((len(train.muscle_names), selected_rank), dtype=np.float64)
    physical_basis[preprocess.kept_indices] = physical_kept_basis
    transform_manifest = None if train.transform is None else train.transform.to_manifest()
    selected_report = rank_reports[selected_rank]
    manifest = {
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
            },
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
    primitive_source_manifest: str | Path | None = None,
    config: SynergyFitConfig | None = None,
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
        regional = load_grouping_json(
            grouping_json,
            muscle_names=train.muscle_names,
            require_complete=True,
        )
        duplicate = sorted(set(groups) & set(regional))
        if duplicate:
            raise ValueError(f"regional labels collide with global labels: {duplicate}")
        groups.update(regional)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
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
    for requested_kind in signal_kinds:
        kind = _canonical_signal_kind(requested_kind)
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
            )
            fitted["artifact_role"] = "global_comparator" if region == "whole_body" else "regional_component"
            reports.append(fitted)
            signal_reports[region] = fitted
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
            )
            reports.append(composite)
            preferred_decoder_artifacts[kind] = {
                "artifact_path": composite["artifact_path"],
                "artifact_fingerprint": composite["artifact_fingerprint"],
                "reason": "regional_composite_is_primary_decoder_basis; whole_body_is_comparator_only",
            }
        elif "whole_body" in signal_reports:
            preferred_decoder_artifacts[kind] = {
                "artifact_path": signal_reports["whole_body"]["artifact_path"],
                "artifact_fingerprint": signal_reports["whole_body"]["artifact_fingerprint"],
                "reason": "global-only mode requested",
            }
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
        "signal_kinds": [_canonical_signal_kind(value) for value in signal_kinds],
        "fit_config": asdict(cfg),
        "preferred_decoder_artifacts": preferred_decoder_artifacts,
        "artifacts": reports,
    }
    excitation_composite = next(
        (
            item
            for item in reports
            if item["signal_kind"] == EXCITATION_SIGNAL_KIND and item["artifact_role"] == "primary_regional_composite"
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
        if excitation_composite is None
        else excitation_composite["promotion_metrics"]
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
        loaded_components.append((indices, start, stop, artifact))
        total_rank = stop
    if total_rank <= 0:
        raise ValueError("regional composite requires at least one fitted component")
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
        # Fail the existing promotion gate when rank selection fell back due to
        # instability, even if the content hashes themselves are internally valid.
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


def _load_explicit_excitation(split: LoadedSynergySplit) -> SynergySignal:
    required = {"teacher_ctrl_physical", "muscle_excitation"}
    missing = sorted(required - set(split.arrays))
    if missing:
        raise ValueError(f"{split.split} excitation fitting is fail-closed and requires raw+unit fields {missing}")
    ctrlrange = np.asarray(split.metadata.get("actuator_ctrlrange"), dtype=np.float64)
    if ctrlrange.shape != (len(split.muscle_names), 2):
        raise ValueError("excitation fitting requires name-aligned metadata.actuator_ctrlrange")
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
    recomputed = ctrl_to_unit_excitation(
        split.arrays["teacher_ctrl_physical"],
        ctrlrange=ctrlrange,
        actuator_names=split.muscle_names,
    )
    stored = np.asarray(split.arrays["muscle_excitation"], dtype=np.float64)
    if stored.shape != recomputed.values.shape or not np.allclose(
        stored,
        recomputed.values,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("stored muscle_excitation differs from explicit raw ctrlrange transform")
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
    finite = np.asarray(result["local_vaf"], dtype=np.float64)
    result["finite_local_vaf_fraction"] = float(np.mean(np.isfinite(finite)))
    return _jsonable(result)


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
        "--primitive-source-manifest",
        default=None,
        help="strict source_manifest.json required for a primitive-only early-control basis",
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=list(range(1, 11)))
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = SynergyFitConfig(
        ranks=tuple(args.ranks),
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
    )
    report = fit_synergy_dataset(
        args.train,
        args.val,
        output_dir=args.output_dir,
        teacher_checkpoint_fingerprint=args.teacher_checkpoint_fingerprint,
        signal_kinds=args.signals,
        mode=args.mode,
        grouping_json=args.grouping_json,
        primitive_source_manifest=args.primitive_source_manifest,
        config=config,
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
