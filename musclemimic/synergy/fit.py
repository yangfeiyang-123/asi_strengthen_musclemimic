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
from musclemimic.distill.action_schema import ordered_schema_hash
from musclemimic.distill.physical import (
    MUSCLE_ACTIVATION_SOURCE,
    UNIT_EXCITATION_TRANSFORM,
    UNIT_INTERVAL_ROUNDOFF_POLICY,
    validate_activation_valid_mask,
    validate_physical_signal_semantics,
    validate_unit_muscle_activation,
)
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
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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
        if not seeds:
            raise ValueError("at least one initialization seed is required")
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
        result = np.asarray(self.arrays["phase_id"], dtype=np.int32)
        if result.ndim != 1:
            raise ValueError("phase_id must have shape [samples]")
        return result

    @property
    def motion_ids(self) -> np.ndarray | None:
        if self.motion_id_field is None:
            return None
        return np.asarray(self.arrays[self.motion_id_field], dtype=np.int64)

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
    train_weights = phase_balanced_weights(train_phases, weights=cfg.phase_weights)
    val_weights = phase_balanced_weights(val_phases, weights=cfg.phase_weights)
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
            train_motion_ids,
            rank=rank,
            seed=cfg.seeds[0] + 30_000 + rank * 100,
            max_iter=cfg.max_iter,
            max_trials=cfg.cross_trial_max_trials,
        )
        local_quantile = _finite_quantile(
            val_metrics["local_vaf"],
            cfg.local_vaf_quantile,
        )
        rejection_reasons = _rank_rejection_reasons(
            val_global_vaf=float(val_metrics["global_vaf"]),
            val_local_quantile=local_quantile,
            initialization=float(init_report["mean_similarity"]),
            split_half=float(split_report["mean_similarity"]),
            bootstrap=float(bootstrap_report["mean_similarity"]),
            cross_trial=cross_trial_report,
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
            "eligible": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
        }

    eligible = [rank for rank in ranks if rank_reports[rank]["eligible"]]
    if eligible:
        selected_rank = min(eligible)
        selection_reason = "smallest_rank_meeting_all_vaf_and_stability_gates"
    else:
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
        "phase_balancing": {
            "weights": {str(key): float(value) for key, value in cfg.phase_weights.items()},
            "sample_weight_application": "multiply_rows_by_sqrt_weight",
            "fit_scope": "train_only",
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
    return {
        "region": str(region),
        "signal_kind": train.signal_kind,
        "selected_rank": selected_rank,
        "selection_reason": selection_reason,
        "artifact_path": str(artifact.path.resolve()),
        "artifact_fingerprint": artifact.fingerprint,
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
    train_motion_uids = set(_unique_ints(train.motion_ids))
    val_motion_uids = set(_unique_ints(val.motion_ids))
    if not train_motion_uids or not val_motion_uids:
        raise ValueError("synergy train/validation motion identity sets must be non-empty")
    motion_overlap = sorted(train_motion_uids & val_motion_uids)
    if motion_overlap:
        raise ValueError(f"synergy train/validation motion leakage detected: {motion_overlap}")
    checkpoint = _resolve_teacher_fingerprint(
        teacher_checkpoint_fingerprint,
        train.metadata,
        val.metadata,
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


def _fit_cross_trial_stability(
    values: np.ndarray,
    motion_ids: np.ndarray | None,
    *,
    rank: int,
    seed: int,
    max_iter: int,
    max_trials: int,
) -> dict[str, Any]:
    if motion_ids is None or int(max_trials) <= 1:
        return {"available": False, "reason": "motion IDs unavailable or disabled", "pair_count": 0}
    ids = np.asarray(motion_ids, dtype=np.int64)
    if ids.shape != (values.shape[0],):
        raise ValueError("motion IDs must match training rows")
    candidates = [(int(uid), int(np.sum(ids == uid))) for uid in np.unique(ids) if int(np.sum(ids == uid)) >= int(rank)]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    selected = candidates[: int(max_trials)]
    bases: list[np.ndarray] = []
    fitted_uids: list[int] = []
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
        fitted_uids.append(uid)
    if len(bases) < 2:
        return {
            "available": False,
            "reason": "fewer than two trials could be fitted at this rank",
            "pair_count": 0,
            "fitted_motion_uids": fitted_uids,
            "skipped": skipped,
        }
    result = cross_trial_stability(bases)
    return {
        "available": True,
        **result,
        "fitted_motion_uids": fitted_uids,
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
    config: SynergyFitConfig,
) -> list[str]:
    reasons: list[str] = []
    if val_global_vaf < config.min_val_global_vaf:
        reasons.append("heldout_global_vaf_below_threshold")
    if val_local_quantile is None or val_local_quantile < config.min_val_local_vaf_quantile:
        reasons.append("heldout_local_vaf_quantile_below_threshold_or_undefined")
    if initialization < config.min_initialization_similarity:
        reasons.append("initialization_stability_below_threshold")
    if split_half < config.min_split_half_similarity:
        reasons.append("split_half_stability_below_threshold")
    if bootstrap < config.min_bootstrap_similarity:
        reasons.append("bootstrap_stability_below_threshold")
    if cross_trial.get("available") and float(cross_trial["mean_similarity"]) < config.min_cross_trial_similarity:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Train dataset directory or one NPZ shard")
    parser.add_argument("--val", required=True, help="Validation dataset directory or one NPZ shard")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-checkpoint-fingerprint", default=None)
    parser.add_argument("--signals", nargs="+", default=["excitation", "activation"])
    parser.add_argument("--mode", choices=["global", "regional", "both"], default="both")
    parser.add_argument("--grouping-json", default=None)
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
