"""Build a versioned, observation-space-only primitive sEMG synergy library.

The ordinary synergy extractor answers whether one dataset admits a useful NMF
description.  A primitive library has a stricter contract:

* every primitive action contributes equal total weight to the shared fit;
* the rank is selected from whole-trial resampling evidence, never from VAF
  alone and never by pinning a caller-provided K;
* the fixed shared basis is projected back onto every source trial so activation
  statistics remain in the unweighted, channel-balanced observation space; and
* the release manifest makes the one-way observation-space boundary explicit.

In particular, this module does not infer or permit a 15/16-channel-to-354
actuator lifting.  Even a formally ready measurement artifact remains disabled
for policy training until a separate, reviewed simulation-side contract exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .models import CHANNEL_NORMALIZATIONS
from .profiles import require_analysis_profile
from .protocols import get_protocol
from .storage import atomic_save_npz, atomic_write_json, dataset_sha256, git_commit_hash, read_json, safe_identifier
from .synergy import channel_scale_diagnostics, fit_nmf_best, match_synergies, vaf_metrics
from .synergy_reuse import (
    basis_geometry,
    project_onto_basis,
)

PRIMITIVE_LIBRARY_SCHEMA_VERSION = "emg_primitive_synergy_library_v2"
PRIMITIVE_LIBRARY_NPZ_NAME = "primitive_synergy_library.npz"
PRIMITIVE_LIBRARY_MANIFEST_NAME = "primitive_synergy_library_manifest.json"
PRIMITIVE_LIBRARY_SCAN_NAME = "primitive_synergy_k_scan.json"
SOURCE_DATASET_DIGEST_VERSION = "emg_synergy_dataset_content_v2"
QC_REVIEW_SCHEMA_VERSION = "emg_primitive_channel_qc_review_v1"


@dataclass(frozen=True)
class PrimitiveLibraryConfig:
    """Numerical and release gates for one primitive-library build."""

    k_min: int = 1
    k_max: int = 8
    n_init: int = 30
    seed: int = 20260720
    split_half_repeats: int = 20
    bootstrap_repeats: int = 20
    initialization_restarts: int = 6
    stability_cosine_threshold: float = 0.80
    split_half_fraction_required: float = 0.50
    bootstrap_median_threshold: float = 0.80
    initialization_minimum_threshold: float = 0.80
    minimum_effective_rank_fraction: float = 0.75
    minimum_fit_global_vaf: float = 0.90
    minimum_fit_local_vaf: float = 0.75
    minimum_fit_local_fraction: float = 0.80
    minimum_heldout_global_vaf: float = 0.75
    minimum_heldout_action_fraction: float = 0.80
    channel_normalization: str = "unit_variance"
    minimum_trials_per_action: int = 4

    def __post_init__(self) -> None:
        if self.k_min < 1 or self.k_max <= self.k_min:
            raise ValueError("Primitive-library K selection requires a scan with 1 <= k_min < k_max")
        if self.n_init < 1:
            raise ValueError("n_init must be >= 1")
        if self.split_half_repeats < 1 or self.bootstrap_repeats < 1:
            raise ValueError("split-half and bootstrap repeats must be >= 1")
        if self.initialization_restarts < 2:
            raise ValueError("initialization_restarts must be >= 2")
        for label, value in (
            ("stability_cosine_threshold", self.stability_cosine_threshold),
            ("split_half_fraction_required", self.split_half_fraction_required),
            ("bootstrap_median_threshold", self.bootstrap_median_threshold),
            ("initialization_minimum_threshold", self.initialization_minimum_threshold),
            ("minimum_effective_rank_fraction", self.minimum_effective_rank_fraction),
            ("minimum_fit_global_vaf", self.minimum_fit_global_vaf),
            ("minimum_fit_local_vaf", self.minimum_fit_local_vaf),
            ("minimum_fit_local_fraction", self.minimum_fit_local_fraction),
            ("minimum_heldout_global_vaf", self.minimum_heldout_global_vaf),
            ("minimum_heldout_action_fraction", self.minimum_heldout_action_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1]")
        if self.channel_normalization not in CHANNEL_NORMALIZATIONS:
            raise ValueError(f"channel_normalization must be one of {CHANNEL_NORMALIZATIONS}")
        if self.minimum_trials_per_action < 4:
            raise ValueError(
                "minimum_trials_per_action must be >= 4 so both stratified halves "
                "contain repeated trials for every action"
            )


def _json_safe(value: Any) -> Any:
    """Return strict-JSON-compatible values for hashing and release manifests."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _canonical_json_sha256(payload: Any) -> str:
    serialised = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _arrays_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Hash named arrays independently of NPZ container timestamps/encoding."""

    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _load_dataset(
    dataset_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Load a synergy dataset and bind every semantic NPZ array to its metadata.

    ``dataset_sha256`` is retained as a verified legacy identity because existing
    datasets use that contract.  The library additionally publishes a v2 digest
    over every required array and the canonical companion metadata, so changing
    a trial boundary, channel label, side, or sample rate cannot preserve the
    source identity merely because ``V`` stayed unchanged.
    """

    dataset_path = Path(dataset_path)
    metadata_path = dataset_path.with_suffix(".json")
    metadata = read_json(metadata_path)
    with np.load(dataset_path, allow_pickle=False) as payload:
        required = (
            "V",
            "channel_ids",
            "muscle_slugs",
            "sides",
            "trial_boundaries",
            "fs_hz",
        )
        missing = sorted(set(required) - set(payload.files))
        if missing:
            raise ValueError(f"Synergy dataset is missing arrays: {missing}")
        arrays = {key: np.asarray(payload[key]).copy() for key in required}

    matrix = arrays["V"]
    boundaries = np.asarray(arrays["trial_boundaries"], dtype=np.int64)
    trials = list(metadata.get("trials", []))
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("Primitive-library V must be finite, nonnegative, and shaped [channels, time]")
    if (
        boundaries.ndim != 1
        or len(boundaries) != len(trials) + 1
        or boundaries[0] != 0
        or boundaries[-1] != matrix.shape[1]
        or np.any(np.diff(boundaries) <= 0)
    ):
        raise ValueError("trial_boundaries must contain one positive-length slice per metadata trial")

    stored_hash = metadata.get("dataset_sha256")
    if not isinstance(stored_hash, str) or len(stored_hash) != 64:
        raise ValueError("Dataset companion JSON must contain dataset_sha256")
    unhashed_metadata = dict(metadata)
    unhashed_metadata.pop("dataset_sha256", None)
    computed_hash = dataset_sha256(matrix, unhashed_metadata)
    if computed_hash != stored_hash:
        raise ValueError("Dataset SHA-256 does not match V and companion metadata")

    if metadata.get("matrix_shape") != list(matrix.shape):
        raise ValueError("Companion matrix_shape does not match V")
    metadata_boundaries = np.asarray(metadata.get("trial_boundaries"), dtype=np.int64)
    if not np.array_equal(boundaries, metadata_boundaries):
        raise ValueError("NPZ trial_boundaries do not match companion metadata")
    expected_lengths = np.asarray(
        [int(trial.get("output_samples", -1)) for trial in trials],
        dtype=np.int64,
    )
    if not np.array_equal(np.diff(boundaries), expected_lengths):
        raise ValueError("trial_boundaries do not match per-trial output_samples")

    fs_array = np.asarray(arrays["fs_hz"])
    if fs_array.size != 1:
        raise ValueError("fs_hz must be a scalar array")
    fs_hz = float(fs_array.reshape(()))
    if (
        not np.isfinite(fs_hz)
        or fs_hz <= 0.0
        or not np.isclose(
            fs_hz,
            float(metadata.get("fs_hz", np.nan)),
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise ValueError("NPZ fs_hz does not match finite positive companion metadata")

    profile = require_analysis_profile(str(metadata.get("profile_id", "")))
    if int(metadata.get("profile_version", -1)) != profile.version:
        raise ValueError("Dataset profile_version does not match its registered profile")
    snapshot = metadata.get("channel_profile_snapshot")
    if snapshot != profile.to_dict():
        raise ValueError("Dataset channel_profile_snapshot does not match its versioned registry profile")
    expected_ids = np.asarray(profile.channel_ids, dtype=np.int64)
    channel_ids = np.asarray(arrays["channel_ids"], dtype=np.int64)
    expected_slugs = np.asarray([channel.muscle_slug for channel in profile.channels])
    expected_sides = np.asarray([channel.side for channel in profile.channels])
    if channel_ids.ndim != 1 or not np.array_equal(channel_ids, expected_ids):
        raise ValueError("Dataset channel_ids do not match the registered profile order")
    if not np.array_equal(np.asarray(arrays["muscle_slugs"]).astype(str), expected_slugs):
        raise ValueError("Dataset muscle_slugs do not match the registered profile order")
    if not np.array_equal(np.asarray(arrays["sides"]).astype(str), expected_sides):
        raise ValueError("Dataset sides do not match the registered profile order")

    actual_actions = sorted({str(trial.get("action_id", "")) for trial in trials})
    if metadata.get("actions") != actual_actions:
        raise ValueError("Companion actions do not match the included trials")
    actual_participants = sorted({str(trial.get("participant_id", "")) for trial in trials})
    if metadata.get("participants") != actual_participants:
        raise ValueError("Companion participants do not match the included trials")
    actual_sessions = sorted({f"{trial.get('participant_id', '')}/{trial.get('session_id', '')}" for trial in trials})
    if metadata.get("sessions") != actual_sessions:
        raise ValueError("Companion sessions do not match the included trials")

    arrays_sha256 = _arrays_sha256(arrays)
    metadata_sha256 = _canonical_json_sha256(unhashed_metadata)
    content_sha256 = _canonical_json_sha256(
        {
            "schema_version": SOURCE_DATASET_DIGEST_VERSION,
            "arrays_sha256": arrays_sha256,
            "metadata_sha256": metadata_sha256,
        }
    )
    integrity = {
        "schema_version": SOURCE_DATASET_DIGEST_VERSION,
        "legacy_dataset_sha256": stored_hash,
        "legacy_dataset_sha256_verified": True,
        "arrays_sha256": arrays_sha256,
        "metadata_sha256": metadata_sha256,
        "content_sha256": content_sha256,
        "npz_file_sha256": _file_sha256(dataset_path),
        "json_file_sha256": _file_sha256(metadata_path),
    }
    return arrays, metadata, integrity


def _trial_slices(boundaries: np.ndarray) -> list[slice]:
    return [slice(int(start), int(stop)) for start, stop in pairwise(boundaries)]


def _action_columns(
    trials: list[dict[str, Any]],
    boundaries: np.ndarray,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, list[int]]]:
    slices = _trial_slices(boundaries)
    trial_indices: dict[str, list[int]] = {}
    seen_trial_ids: set[str] = set()
    for index, trial in enumerate(trials):
        action = str(trial.get("action_id", ""))
        if not action:
            raise ValueError(f"Dataset trial {index} has no action_id")
        trial_id = str(trial.get("trial_id", ""))
        if not trial_id or trial_id in seen_trial_ids:
            raise ValueError(f"Dataset trial IDs must be non-empty and unique; got {trial_id!r}")
        seen_trial_ids.add(trial_id)
        trial_indices.setdefault(action, []).append(index)
    actions = sorted(trial_indices)
    columns = {
        action: np.concatenate(
            [np.arange(slices[index].start, slices[index].stop, dtype=np.int64) for index in trial_indices[action]]
        )
        for action in actions
    }
    return actions, columns, trial_indices


def action_balanced_channel_scale(
    matrix: np.ndarray,
    action_columns: dict[str, np.ndarray],
    mode: str,
) -> np.ndarray:
    """Channel divisor under a distribution that gives every action equal mass."""

    values = np.asarray(matrix, dtype=np.float64)
    if mode not in CHANNEL_NORMALIZATIONS:
        raise ValueError(f"channel_normalization must be one of {CHANNEL_NORMALIZATIONS}")
    if not action_columns:
        raise ValueError("At least one primitive action is required")
    if mode == "none":
        scale = np.ones(values.shape[0], dtype=np.float64)
    elif mode == "unit_max":
        scale = np.max(values, axis=1)
    else:
        action_means = np.stack([np.mean(values[:, columns], axis=1) for columns in action_columns.values()])
        equal_action_mean = np.mean(action_means, axis=0)
        equal_action_variance = np.mean(
            np.stack(
                [
                    np.mean((values[:, columns] - equal_action_mean[:, None]) ** 2, axis=1)
                    for columns in action_columns.values()
                ]
            ),
            axis=0,
        )
        scale = np.sqrt(np.maximum(equal_action_variance, 0.0))
    if not np.all(np.isfinite(scale)):
        raise ValueError("Action-balanced channel scaling produced a non-finite divisor")
    silent = scale <= np.finfo(np.float64).eps
    if np.any(silent) and mode != "none":
        raise ValueError(
            f"Channels {np.flatnonzero(silent).tolist()} are numerically silent and cannot be action-balanced"
        )
    return np.maximum(scale, np.finfo(np.float64).eps)


def action_balance_weights(
    total_columns: int,
    action_columns: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    """Column multipliers that give every action equal total squared weight."""

    if total_columns < 1 or not action_columns:
        raise ValueError("A non-empty matrix and at least one action are required")
    target_squared_mass = total_columns / len(action_columns)
    weights = np.empty(total_columns, dtype=np.float64)
    multipliers: dict[str, float] = {}
    assigned = np.zeros(total_columns, dtype=bool)
    for action, columns in action_columns.items():
        if columns.size == 0:
            raise ValueError(f"Primitive action {action!r} has no samples")
        multiplier = float(np.sqrt(target_squared_mass / columns.size))
        weights[columns] = multiplier
        assigned[columns] = True
        multipliers[action] = multiplier
    if not np.all(assigned):
        raise ValueError("Every dataset column must belong to exactly one primitive action")
    return weights, multipliers


def _insufficient_action_trials(
    action_trial_indices: dict[str, list[int]],
    minimum: int,
) -> dict[str, int]:
    return {action: len(indices) for action, indices in action_trial_indices.items() if len(indices) < minimum}


def _resampled_action_balanced_fit(
    balanced_matrix: np.ndarray,
    boundaries: np.ndarray,
    trials: list[dict[str, Any]],
    sampled_trial_indices: dict[str, list[int]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Concatenate sampled whole trials and restore equal action objective mass."""

    slices = _trial_slices(boundaries)
    parts: list[np.ndarray] = []
    local_action_columns: dict[str, np.ndarray] = {}
    audit_actions: dict[str, Any] = {}
    offset = 0
    for action in sorted(sampled_trial_indices):
        indices = [int(index) for index in sampled_trial_indices[action]]
        if not indices:
            raise ValueError(f"Stratified resample omitted action {action!r}")
        action_parts = [balanced_matrix[:, slices[index]] for index in indices]
        action_matrix = np.concatenate(action_parts, axis=1)
        columns = np.arange(offset, offset + action_matrix.shape[1], dtype=np.int64)
        local_action_columns[action] = columns
        parts.append(action_matrix)
        offset += action_matrix.shape[1]
        audit_actions[action] = {
            "draw_count": len(indices),
            "source_trial_indices": indices,
            "trial_ids": [str(trials[index]["trial_id"]) for index in indices],
            "columns": int(action_matrix.shape[1]),
        }
    resampled = np.concatenate(parts, axis=1)
    weights, multipliers = action_balance_weights(resampled.shape[1], local_action_columns)
    for action, columns in local_action_columns.items():
        audit_actions[action]["fit_column_multiplier"] = multipliers[action]
        audit_actions[action]["total_squared_fit_weight"] = float(np.sum(weights[columns] ** 2))
    return resampled * weights[None, :], {
        "actions": audit_actions,
        "equal_total_squared_fit_weight_per_action": True,
    }


def _heldout_action_vaf(
    basis: np.ndarray,
    balanced_matrix: np.ndarray,
    boundaries: np.ndarray,
    sampled_trial_indices: dict[str, list[int]],
    local_vaf_threshold: float,
) -> dict[str, Any]:
    slices = _trial_slices(boundaries)
    result: dict[str, Any] = {}
    for action in sorted(sampled_trial_indices):
        columns = np.concatenate(
            [
                np.arange(slices[index].start, slices[index].stop, dtype=np.int64)
                for index in sampled_trial_indices[action]
            ]
        )
        values = balanced_matrix[:, columns]
        _, reconstruction = project_onto_basis(basis, values)
        global_vaf, local_vaf, _ = vaf_metrics(values, reconstruction)
        result[action] = {
            "global_vaf": global_vaf,
            "local_vaf": local_vaf.tolist(),
            "local_vaf_threshold": local_vaf_threshold,
            "local_vaf_fraction": float(np.mean(local_vaf >= local_vaf_threshold)),
        }
    return result


def _action_stratified_split_half(
    balanced_matrix: np.ndarray,
    boundaries: np.ndarray,
    trials: list[dict[str, Any]],
    action_trial_indices: dict[str, list[int]],
    k: int,
    config: PrimitiveLibraryConfig,
    seed: int,
) -> dict[str, Any]:
    insufficient = _insufficient_action_trials(
        action_trial_indices,
        config.minimum_trials_per_action,
    )
    if insufficient:
        return {
            "available": False,
            "reason": "action_stratified_split_requires_minimum_trials_per_action",
            "minimum_trials_per_action": config.minimum_trials_per_action,
            "insufficient_actions": insufficient,
            "cosine_threshold": config.stability_cosine_threshold,
            "heldout": {"available": False, "reason": "split_half_unavailable"},
        }

    generator = np.random.default_rng(seed)
    runs: list[dict[str, Any]] = []
    heldout_scores: dict[str, list[float]] = {action: [] for action in action_trial_indices}
    for repeat in range(config.split_half_repeats):
        group_a: dict[str, list[int]] = {}
        group_b: dict[str, list[int]] = {}
        for action in sorted(action_trial_indices):
            shuffled = generator.permutation(action_trial_indices[action]).tolist()
            half = len(shuffled) // 2
            group_a[action] = [int(value) for value in shuffled[:half]]
            group_b[action] = [int(value) for value in shuffled[half:]]
        fit_a, audit_a = _resampled_action_balanced_fit(balanced_matrix, boundaries, trials, group_a)
        fit_b, audit_b = _resampled_action_balanced_fit(balanced_matrix, boundaries, trials, group_b)
        if fit_a.shape[1] < k or fit_b.shape[1] < k:
            continue
        basis_a, _, _ = fit_nmf_best(
            fit_a,
            k,
            config.n_init,
            seed + 10000 + repeat * 17,
        )
        basis_b, _, _ = fit_nmf_best(
            fit_b,
            k,
            config.n_init,
            seed + 20000 + repeat * 17,
        )
        match = match_synergies(basis_a, basis_b)
        a_to_b = _heldout_action_vaf(
            basis_a,
            balanced_matrix,
            boundaries,
            group_b,
            config.minimum_fit_local_vaf,
        )
        b_to_a = _heldout_action_vaf(
            basis_b,
            balanced_matrix,
            boundaries,
            group_a,
            config.minimum_fit_local_vaf,
        )
        for action in heldout_scores:
            heldout_scores[action].extend([a_to_b[action]["global_vaf"], b_to_a[action]["global_vaf"]])
        runs.append(
            {
                "repeat_index": repeat,
                "mean_cosine_similarity": match["mean_cosine_similarity"],
                "minimum_cosine_similarity": match["minimum_cosine_similarity"],
                "cosine_similarities": match["cosine_similarities"],
                "groups": {"a": audit_a, "b": audit_b},
                "heldout": {"a_to_b": a_to_b, "b_to_a": b_to_a},
            }
        )
    if not runs:
        return {
            "available": False,
            "reason": "insufficient_samples_per_stratified_half",
            "cosine_threshold": config.stability_cosine_threshold,
            "heldout": {"available": False, "reason": "split_half_unavailable"},
        }

    minima = np.asarray([run["minimum_cosine_similarity"] for run in runs], dtype=np.float64)
    by_action = {
        action: {
            "scores": scores,
            "median_global_vaf": float(np.median(scores)),
            "minimum_global_vaf": float(np.min(scores)),
            "passes_threshold": bool(np.median(scores) >= config.minimum_heldout_global_vaf),
        }
        for action, scores in heldout_scores.items()
    }
    heldout_action_fraction = float(np.mean([entry["passes_threshold"] for entry in by_action.values()]))
    return {
        "available": True,
        "requested_repeats": config.split_half_repeats,
        "repeats": len(runs),
        "skipped_repeats": config.split_half_repeats - len(runs),
        "partition": "action-stratified random halves of whole trials",
        "resample_weighting": "equal total squared fit weight per action rebuilt in every half",
        "cosine_threshold": config.stability_cosine_threshold,
        "mean_cosine_similarity": float(np.mean([run["mean_cosine_similarity"] for run in runs])),
        "median_minimum_cosine_similarity": float(np.median(minima)),
        "worst_minimum_cosine_similarity": float(np.min(minima)),
        "fraction_of_repeats_all_synergies_ge_threshold": float(np.mean(minima >= config.stability_cosine_threshold)),
        "runs": runs,
        "heldout": {
            "available": True,
            "global_vaf_threshold": config.minimum_heldout_global_vaf,
            "required_action_fraction": config.minimum_heldout_action_fraction,
            "action_fraction_meeting_threshold": heldout_action_fraction,
            "by_action": by_action,
        },
    }


def _action_stratified_bootstrap(
    balanced_matrix: np.ndarray,
    boundaries: np.ndarray,
    trials: list[dict[str, Any]],
    action_trial_indices: dict[str, list[int]],
    reference_basis: np.ndarray,
    k: int,
    config: PrimitiveLibraryConfig,
    seed: int,
) -> dict[str, Any]:
    insufficient = _insufficient_action_trials(
        action_trial_indices,
        config.minimum_trials_per_action,
    )
    if insufficient:
        return {
            "available": False,
            "reason": "action_stratified_bootstrap_requires_minimum_trials_per_action",
            "minimum_trials_per_action": config.minimum_trials_per_action,
            "insufficient_actions": insufficient,
        }
    generator = np.random.default_rng(seed)
    runs: list[dict[str, Any]] = []
    for repeat in range(config.bootstrap_repeats):
        sampled = {
            action: [int(value) for value in generator.choice(indices, size=len(indices), replace=True).tolist()]
            for action, indices in sorted(action_trial_indices.items())
        }
        fit_matrix, audit = _resampled_action_balanced_fit(
            balanced_matrix,
            boundaries,
            trials,
            sampled,
        )
        if fit_matrix.shape[1] < k:
            continue
        fitted, _, _ = fit_nmf_best(
            fit_matrix,
            k,
            config.n_init,
            seed + 5000 + repeat * 19,
        )
        match = match_synergies(reference_basis, fitted)
        runs.append(
            {
                "repeat_index": repeat,
                "mean_cosine_similarity": match["mean_cosine_similarity"],
                "minimum_cosine_similarity": match["minimum_cosine_similarity"],
                "cosine_similarities": match["cosine_similarities"],
                "draw": audit,
            }
        )
    if not runs:
        return {"available": False, "reason": "insufficient_samples_per_stratified_resample"}
    minima = np.asarray([run["minimum_cosine_similarity"] for run in runs], dtype=np.float64)
    return {
        "available": True,
        "requested_repeats": config.bootstrap_repeats,
        "repeats": len(runs),
        "skipped_repeats": config.bootstrap_repeats - len(runs),
        "resample": "whole trials with replacement independently within every action",
        "resample_weighting": "equal total squared fit weight per action rebuilt in every draw",
        "cosine_threshold": config.stability_cosine_threshold,
        "mean_cosine_similarity": float(np.mean([run["mean_cosine_similarity"] for run in runs])),
        "median_minimum_cosine_similarity": float(np.median(minima)),
        "minimum_component_cosine_similarity": float(np.min(minima)),
        "fraction_of_repeats_all_synergies_ge_threshold": float(np.mean(minima >= config.stability_cosine_threshold)),
        "runs": runs,
    }


def _initialization_stability_per_component(
    fit_matrix: np.ndarray,
    k: int,
    config: PrimitiveLibraryConfig,
    seed: int,
) -> dict[str, Any]:
    bases = [
        fit_nmf_best(
            fit_matrix,
            k,
            config.n_init,
            seed + restart * 104729,
        )[0]
        for restart in range(config.initialization_restarts)
    ]
    pairs: list[dict[str, Any]] = []
    for first in range(len(bases)):
        for second in range(first + 1, len(bases)):
            match = match_synergies(bases[first], bases[second])
            pairs.append(
                {
                    "restart_indices": [first, second],
                    "mean_cosine_similarity": match["mean_cosine_similarity"],
                    "minimum_cosine_similarity": match["minimum_cosine_similarity"],
                    "cosine_similarities": match["cosine_similarities"],
                }
            )
    minima = np.asarray([pair["minimum_cosine_similarity"] for pair in pairs], dtype=np.float64)
    return {
        "restarts": config.initialization_restarts,
        "pair_count": len(pairs),
        "cosine_threshold": config.initialization_minimum_threshold,
        "mean_pair_cosine_similarity": float(np.mean([pair["mean_cosine_similarity"] for pair in pairs])),
        "minimum_component_cosine_similarity": float(np.min(minima)),
        "pairs": pairs,
    }


def _scan_k(
    balanced_matrix: np.ndarray,
    fit_matrix: np.ndarray,
    boundaries: np.ndarray,
    trials: list[dict[str, Any]],
    action_trial_indices: dict[str, list[int]],
    config: PrimitiveLibraryConfig,
) -> tuple[list[dict[str, Any]], int, str]:
    max_components = min(config.k_max, fit_matrix.shape[0], fit_matrix.shape[1])
    candidate_ks = list(range(config.k_min, max_components + 1))
    if len(candidate_ks) < 2:
        raise ValueError("Primitive-library rank must be selected from at least two feasible K values")

    rows: list[dict[str, Any]] = []
    for k in candidate_ks:
        basis, _, fit_metrics = fit_nmf_best(fit_matrix, k, config.n_init, config.seed + k * 1009)
        split_half = _action_stratified_split_half(
            balanced_matrix,
            boundaries,
            trials,
            action_trial_indices,
            k,
            config,
            config.seed + k * 2003,
        )
        initialization = _initialization_stability_per_component(
            fit_matrix,
            k,
            config,
            config.seed + k * 3001,
        )
        bootstrap = _action_stratified_bootstrap(
            balanced_matrix,
            boundaries,
            trials,
            action_trial_indices,
            basis,
            k,
            config,
            config.seed + k * 4001,
        )
        geometry = basis_geometry(basis)
        local_vaf = np.asarray(fit_metrics["local_vaf"], dtype=np.float64)
        fit_local_fraction = float(np.mean(local_vaf >= config.minimum_fit_local_vaf))
        heldout = split_half.get("heldout", {})
        stability_gates = {
            "whole_trial_action_stratified_split_half_available": bool(split_half.get("available")),
            "split_half_median_minimum_component_cosine": bool(
                split_half.get("median_minimum_cosine_similarity", -np.inf) >= config.stability_cosine_threshold
            ),
            "split_half_success_fraction": bool(
                split_half.get("fraction_of_repeats_all_synergies_ge_threshold", -np.inf)
                >= config.split_half_fraction_required
            ),
            "initialization_minimum_component_cosine": bool(
                initialization["minimum_component_cosine_similarity"] >= config.initialization_minimum_threshold
            ),
            "whole_trial_action_stratified_bootstrap_available": bool(bootstrap.get("available")),
            "bootstrap_median_minimum_component_cosine": bool(
                bootstrap.get("median_minimum_cosine_similarity", -np.inf) >= config.bootstrap_median_threshold
            ),
            "effective_rank_fraction": bool(
                geometry["effective_rank_fraction"] >= config.minimum_effective_rank_fraction
            ),
        }
        adequacy_gates = {
            "action_balanced_fit_global_vaf": bool(fit_metrics["global_vaf"] >= config.minimum_fit_global_vaf),
            "action_balanced_fit_local_vaf_fraction": bool(fit_local_fraction >= config.minimum_fit_local_fraction),
            "action_stratified_heldout_available": bool(heldout.get("available")),
            "action_stratified_heldout_action_fraction": bool(
                heldout.get("action_fraction_meeting_threshold", -np.inf) >= config.minimum_heldout_action_fraction
            ),
        }
        stability_pass = all(stability_gates.values())
        adequacy_pass = all(adequacy_gates.values())
        rows.append(
            {
                "k": k,
                "fit_metrics": fit_metrics,
                "fit_local_vaf_threshold": config.minimum_fit_local_vaf,
                "fit_local_vaf_fraction": fit_local_fraction,
                "split_half": split_half,
                "initialization": initialization,
                "bootstrap": bootstrap,
                "basis_geometry": geometry,
                "stability_gates": stability_gates,
                "adequacy_gates": adequacy_gates,
                "stability_pass": stability_pass,
                "adequacy_pass": adequacy_pass,
                "selection_pass": stability_pass and adequacy_pass,
            }
        )

    passed = [row for row in rows if row["selection_pass"]]
    if passed:
        return (
            rows,
            min(int(row["k"]) for row in passed),
            "smallest_k_passing_all_adequacy_and_stability_gates",
        )

    def candidate_score(row: dict[str, Any]) -> tuple[float, ...]:
        all_gates = {**row["adequacy_gates"], **row["stability_gates"]}
        split = row["split_half"]
        bootstrap = row["bootstrap"]
        return (
            float(sum(bool(value) for value in all_gates.values())),
            float(row["fit_metrics"]["global_vaf"]),
            float(split.get("heldout", {}).get("action_fraction_meeting_threshold", -1.0)),
            float(split.get("fraction_of_repeats_all_synergies_ge_threshold", -1.0)),
            float(bootstrap.get("median_minimum_cosine_similarity", -1.0)),
            -float(row["k"]),
        )

    selected = max(rows, key=candidate_score)
    return rows, int(selected["k"]), "highest_joint_evidence_candidate_no_k_passed"


def _activation_statistics(
    basis: np.ndarray,
    balanced_matrix: np.ndarray,
    trials: list[dict[str, Any]],
    boundaries: np.ndarray,
    action_trial_indices: dict[str, list[int]],
    time_normalize_samples: int | None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    coefficients, reconstruction = project_onto_basis(basis, balanced_matrix)
    slices = _trial_slices(boundaries)
    action_statistics: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for action_index, action in enumerate(sorted(action_trial_indices)):
        trial_indices = action_trial_indices[action]
        columns = np.concatenate(
            [np.arange(slices[index].start, slices[index].stop, dtype=np.int64) for index in trial_indices]
        )
        action_coefficients = coefficients[:, columns]
        total_recruitment = float(np.sum(action_coefficients)) or 1.0
        action_vaf, local_vaf, _ = vaf_metrics(balanced_matrix[:, columns], reconstruction[:, columns])
        key_prefix = f"action_{action_index:03d}"
        entry: dict[str, Any] = {
            "action_index": action_index,
            "trial_count": len(trial_indices),
            "trial_ids": [str(trials[index]["trial_id"]) for index in trial_indices],
            "columns": int(columns.size),
            "balanced_projection_global_vaf": action_vaf,
            "balanced_projection_local_vaf": local_vaf.tolist(),
            "mean_coefficient": np.mean(action_coefficients, axis=1).tolist(),
            "recruitment_share": (np.sum(action_coefficients, axis=1) / total_recruitment).tolist(),
            "peak_coefficient": np.max(action_coefficients, axis=1).tolist(),
        }
        trial_profiles = [coefficients[:, slices[index]] for index in trial_indices]
        if (
            time_normalize_samples is not None
            and time_normalize_samples >= 2
            and all(profile.shape[1] == time_normalize_samples for profile in trial_profiles)
        ):
            stacked = np.stack(trial_profiles, axis=0)
            mean_profile = np.mean(stacked, axis=0)
            std_profile = np.std(stacked, axis=0)
            mean_key = f"{key_prefix}_H_mean"
            std_key = f"{key_prefix}_H_std"
            arrays[mean_key] = mean_profile.astype(np.float32)
            arrays[std_key] = std_profile.astype(np.float32)
            entry["activation_array_keys"] = {"mean": mean_key, "std": std_key}
            entry["activation_shape"] = list(mean_profile.shape)
            entry["peak_phase_percent"] = (
                np.argmax(mean_profile, axis=1) * 100.0 / max(1, time_normalize_samples - 1)
            ).tolist()
        action_statistics[action] = entry
    return coefficients, action_statistics, arrays


def _path_identity(path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    try:
        repo_relative: str | None = str(resolved.relative_to(repo_root))
    except ValueError:
        repo_relative = None
    return {"resolved": str(resolved), "repo_relative": repo_relative}


def _source_code_provenance() -> dict[str, Any]:
    """Bind the build to source bytes and fail formal release on dirty code."""

    repo_root = Path(__file__).resolve().parents[2]
    source_paths = [
        Path(__file__).with_name("__init__.py").resolve(),
        Path(__file__).resolve(),
        Path(__file__).with_name("cli.py").resolve(),
        Path(__file__).with_name("dataset.py").resolve(),
        Path(__file__).with_name("models.py").resolve(),
        Path(__file__).with_name("profiles.py").resolve(),
        Path(__file__).with_name("protocols.py").resolve(),
        Path(__file__).with_name("storage.py").resolve(),
        Path(__file__).with_name("synergy.py").resolve(),
        Path(__file__).with_name("synergy_reuse.py").resolve(),
        (repo_root / "jidian_measurement" / "pyproject.toml").resolve(),
    ]
    lock_path = repo_root / "jidian_measurement" / "uv.lock"
    if lock_path.exists():
        source_paths.append(lock_path.resolve())
    source_files = {
        str(path.relative_to(repo_root)): {
            "sha256": _file_sha256(path),
            "num_bytes": path.stat().st_size,
        }
        for path in source_paths
    }
    relative_paths = list(source_files)
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *relative_paths],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status_entries = [line for line in result.stdout.splitlines() if line.strip()]
        git_available = True
    except (OSError, subprocess.SubprocessError):
        status_entries = ["git_status_unavailable"]
        git_available = False
    commit = git_commit_hash(repo_root)
    return {
        "repo_root": str(repo_root),
        "git_available": git_available,
        "git_commit_hash": commit,
        "dirty": bool(status_entries),
        "git_status_entries": status_entries,
        "source_files": source_files,
        "source_bundle_sha256": _canonical_json_sha256(source_files),
        "formal_reproducible": bool(git_available and commit and not status_entries),
    }


def _channel_diagnostics_sha256(diagnostics: dict[str, Any]) -> str:
    return _canonical_json_sha256(
        {
            "schema_version": QC_REVIEW_SCHEMA_VERSION,
            "channel_scale_diagnostics": diagnostics,
        }
    )


def _load_qc_review(
    review_path: Path | None,
    *,
    source_content_sha256: str,
    diagnostics: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    diagnostics_sha256 = _channel_diagnostics_sha256(diagnostics)
    if review_path is None:
        return {
            "status": "missing",
            "required_for_formal_release": True,
            "channel_diagnostics_sha256": diagnostics_sha256,
        }
    review_path = Path(review_path).resolve()
    payload = read_json(review_path)
    if payload.get("schema_version") != QC_REVIEW_SCHEMA_VERSION:
        raise ValueError(f"QC review must use schema_version={QC_REVIEW_SCHEMA_VERSION}")
    status = str(payload.get("status", ""))
    if status not in {"approved", "rejected"}:
        raise ValueError("QC review status must be approved or rejected")
    review_id = safe_identifier(str(payload.get("review_id", "")), "review_id")
    reviewer_id = safe_identifier(str(payload.get("reviewer_id", "")), "reviewer_id")
    reviewed_time = str(payload.get("reviewed_time", ""))
    try:
        datetime.fromisoformat(reviewed_time.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("QC review reviewed_time must be ISO-8601") from exc
    if payload.get("source_content_sha256") != source_content_sha256:
        raise ValueError("QC review source_content_sha256 does not match the source dataset")
    if payload.get("channel_diagnostics_sha256") != diagnostics_sha256:
        raise ValueError("QC review channel_diagnostics_sha256 does not match current diagnostics")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("QC review must bind an evidence file")
    evidence_path_value = evidence.get("path")
    evidence_sha256 = str(evidence.get("sha256", ""))
    if not evidence_path_value or not _is_sha256(evidence_sha256):
        raise ValueError("QC review evidence must provide path and SHA-256")
    evidence_path = Path(str(evidence_path_value))
    if not evidence_path.is_absolute():
        evidence_path = review_path.parent / evidence_path
    evidence_path = evidence_path.resolve()
    if not evidence_path.is_file() or _file_sha256(evidence_path) != evidence_sha256:
        raise ValueError("QC review evidence file is missing or its SHA-256 does not match")
    return {
        "status": status,
        "required_for_formal_release": True,
        "review_id": review_id,
        "reviewer_id": reviewer_id,
        "reviewed_time": reviewed_time,
        "channel_diagnostics_sha256": diagnostics_sha256,
        "source_content_sha256": source_content_sha256,
        "review_manifest": {
            **_path_identity(review_path, repo_root),
            "sha256": _file_sha256(review_path),
        },
        "evidence": {
            **_path_identity(evidence_path, repo_root),
            "sha256": evidence_sha256,
        },
    }


def _readiness_report(
    metadata: dict[str, Any],
    actions: list[str],
    action_trial_indices: dict[str, list[int]],
    expected_actions: list[str],
    selected_row: dict[str, Any],
    diagnostics: dict[str, Any],
    qc_review: dict[str, Any],
    source_provenance: dict[str, Any],
    config: PrimitiveLibraryConfig,
) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    missing_actions = sorted(set(expected_actions) - set(actions))
    if missing_actions:
        blockers.append({"category": "data", "code": "missing_primitive_actions", "actions": missing_actions})

    gated_actions = sorted(set(expected_actions) | set(actions))
    insufficient = {
        action: len(action_trial_indices.get(action, []))
        for action in gated_actions
        if len(action_trial_indices.get(action, [])) < config.minimum_trials_per_action
    }
    if insufficient:
        blockers.append(
            {
                "category": "qc_or_collection",
                "code": "insufficient_analysis_ready_trials",
                "minimum_required": config.minimum_trials_per_action,
                "counts": insufficient,
            }
        )

    exploratory_trials = [
        str(trial.get("trial_id"))
        for trial in metadata.get("trials", [])
        if trial.get("crop_is_exploratory") or trial.get("movement_start_annotation") is None
    ]
    if metadata.get("exploratory") or metadata.get("crop_mode") != "annotated_movement_events" or exploratory_trials:
        blockers.append(
            {
                "category": "event",
                "code": "formal_event_crop_not_proven",
                "crop_mode": metadata.get("crop_mode"),
                "trial_ids": exploratory_trials,
            }
        )

    if not metadata.get("only_valid") or not metadata.get("only_valid_includes_preprocessing_qc"):
        blockers.append(
            {
                "category": "qc",
                "code": "validity_and_preprocessing_qc_gate_not_proven",
                "only_valid": metadata.get("only_valid"),
                "only_valid_includes_preprocessing_qc": metadata.get("only_valid_includes_preprocessing_qc"),
            }
        )
    nonready = [
        str(trial.get("trial_id"))
        for trial in metadata.get("trials", [])
        if trial.get("valid_for_analysis") is not True or trial.get("preprocessing_analysis_ready") is not True
    ]
    if nonready:
        blockers.append({"category": "qc", "code": "included_trial_not_analysis_ready", "trial_ids": nonready})

    suspicious_channels = sorted(
        set(diagnostics.get("channels_far_below_median_scale", []))
        | set(diagnostics.get("channels_with_outlier_driven_peak", []))
    )
    if qc_review.get("status") == "missing":
        blockers.append(
            {
                "category": "qc",
                "code": "independent_channel_qc_review_required",
                "zero_based_channel_indices": suspicious_channels,
                "channel_diagnostics_sha256": qc_review.get("channel_diagnostics_sha256"),
            }
        )
    elif qc_review.get("status") == "rejected":
        blockers.append(
            {
                "category": "qc",
                "code": "independent_channel_qc_review_rejected",
                "review_id": qc_review.get("review_id"),
            }
        )
    if not source_provenance.get("formal_reproducible"):
        blockers.append(
            {
                "category": "provenance",
                "code": "source_tree_not_clean_and_reproducible",
                "git_available": source_provenance.get("git_available"),
                "git_commit_hash": source_provenance.get("git_commit_hash"),
                "git_status_entries": source_provenance.get("git_status_entries", []),
            }
        )
    if not selected_row["selection_pass"]:
        blockers.append(
            {
                "category": "rank_selection",
                "code": "no_k_passed_all_adequacy_and_stability_gates",
                "selected_candidate_k": selected_row["k"],
                "failed_stability_gates": [
                    name for name, passed in selected_row["stability_gates"].items() if not passed
                ],
                "failed_adequacy_gates": [
                    name for name, passed in selected_row["adequacy_gates"].items() if not passed
                ],
            }
        )
    return {
        "formal_ready": not blockers,
        "release_tier": "formal_measurement" if not blockers else "exploratory_candidate",
        "missing_actions": missing_actions,
        "analysis_ready_trial_counts": {action: len(action_trial_indices.get(action, [])) for action in gated_actions},
        "blockers": blockers,
    }


def _assert_unique_library_id(
    parent: Path,
    library_id: str,
    *,
    ignore: Path | None = None,
) -> None:
    for candidate in parent.iterdir() if parent.exists() else ():
        if ignore is not None and candidate.resolve() == ignore.resolve():
            continue
        manifest_path = candidate / PRIMITIVE_LIBRARY_MANIFEST_NAME
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        if manifest.get("library_id") == library_id:
            raise FileExistsError(f"Primitive library_id {library_id!r} is immutable and already exists at {candidate}")


def verify_primitive_synergy_library(output_dir: Path) -> dict[str, Any]:
    """Verify a staged or published v2 library without trusting embedded hashes."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / PRIMITIVE_LIBRARY_MANIFEST_NAME
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != PRIMITIVE_LIBRARY_SCHEMA_VERSION:
        raise ValueError("Primitive library schema_version is not supported by this verifier")
    claimed_manifest_sha = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if claimed_manifest_sha != _canonical_json_sha256(unsigned):
        raise ValueError("Primitive library manifest_sha256 is invalid")
    artifact_contract = manifest.get("artifacts", {})
    library_path = output_dir / str(artifact_contract.get("library_npz", ""))
    scan_path = output_dir / str(artifact_contract.get("k_scan_json", ""))
    if not library_path.is_file() or _file_sha256(library_path) != artifact_contract.get("library_npz_sha256"):
        raise ValueError("Primitive library NPZ is missing or its SHA-256 is invalid")
    if not scan_path.is_file() or _file_sha256(scan_path) != artifact_contract.get("k_scan_json_sha256"):
        raise ValueError("Primitive library K scan is missing or its SHA-256 is invalid")
    with np.load(library_path, allow_pickle=False) as payload:
        basis_keys = (
            "W",
            "W_recorded_units",
            "channel_scale",
            "channel_ids",
            "muscle_slugs",
            "sides",
            "profile_id",
            "profile_version",
            "ordered_channel_schema_sha256",
            "k",
        )
        recomputed_basis_sha256 = _arrays_sha256({key: np.asarray(payload[key]) for key in basis_keys})
        if recomputed_basis_sha256 != manifest["basis"]["basis_sha256"]:
            raise ValueError("Primitive basis arrays do not reproduce basis_sha256")
        if str(payload["basis_sha256"]) != manifest["basis"]["basis_sha256"]:
            raise ValueError("Embedded basis_sha256 does not match the manifest")
        if str(payload["source_content_sha256"]) != manifest["source_dataset"]["content_sha256"]:
            raise ValueError("Embedded source_content_sha256 does not match the manifest")
        if str(payload["ordered_channel_schema_sha256"]) != manifest["basis"]["ordered_channel_schema_sha256"]:
            raise ValueError("Embedded ordered channel schema hash does not match the manifest")
    for label in ("npz", "json"):
        source_entry = manifest["source_dataset"][label]
        source_path = Path(source_entry["resolved"])
        if not source_path.is_file() or _file_sha256(source_path) != source_entry["sha256"]:
            raise ValueError(f"Source dataset {label} is missing or its SHA-256 is invalid")
    qc_review = manifest.get("quality", {}).get("independent_qc_review", {})
    if qc_review.get("status") in {"approved", "rejected"}:
        for label in ("review_manifest", "evidence"):
            entry = qc_review[label]
            path = Path(entry["resolved"])
            if not path.is_file() or _file_sha256(path) != entry["sha256"]:
                raise ValueError(f"Independent QC {label} is missing or its SHA-256 is invalid")
    return {
        "manifest_sha256": claimed_manifest_sha,
        "library_npz_sha256": artifact_contract["library_npz_sha256"],
        "k_scan_json_sha256": artifact_contract["k_scan_json_sha256"],
    }


def build_primitive_synergy_library(
    dataset_path: Path,
    output_dir: Path,
    *,
    library_id: str | None = None,
    required_action_ids: list[str] | None = None,
    config: PrimitiveLibraryConfig | None = None,
    qc_review_manifest: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Build a library in a sibling staging directory and publish it once.

    Published library IDs and directories are immutable.  ``overwrite`` remains
    in the Python signature only to fail older callers explicitly; a changed
    dataset, contract, review, or implementation requires a new output directory
    and a new ``library_id``.
    """

    if overwrite:
        raise ValueError("Primitive libraries are immutable; overwrite is forbidden, create a new version")
    config = config or PrimitiveLibraryConfig()
    dataset_path = Path(dataset_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Primitive library output already exists and is immutable: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    library_id = safe_identifier(library_id or output_dir.name, "library_id")
    _assert_unique_library_id(output_dir.parent, library_id)
    if required_action_ids is not None and not required_action_ids:
        raise ValueError("required_action_ids cannot be an empty explicit library contract")

    source_arrays, metadata, source_integrity = _load_dataset(dataset_path)
    recorded_values = np.asarray(source_arrays["V"], dtype=np.float64)
    boundaries = np.asarray(source_arrays["trial_boundaries"], dtype=np.int64)
    trials = list(metadata["trials"])
    actions, action_columns, action_trial_indices = _action_columns(trials, boundaries)

    protocol = get_protocol(str(metadata["protocol_id"]))
    primitive_protocol_actions = [action.action_id for action in protocol.actions if action.category == "primitive"]
    categories = {action.action_id: action.category for action in protocol.actions}
    nonprimitive = sorted(action for action in actions if categories.get(action) != "primitive")
    if nonprimitive:
        raise ValueError(f"Primitive library refuses complete/unknown actions: {nonprimitive}")
    if metadata.get("scope") != "primitive" and required_action_ids is None:
        expected_actions = actions
        library_scope = "explicit_actions"
    elif required_action_ids is not None:
        expected_actions = sorted(set(required_action_ids))
        invalid_required = sorted(action for action in expected_actions if categories.get(action) != "primitive")
        if invalid_required:
            raise ValueError(f"required_action_ids contains complete/unknown actions: {invalid_required}")
        library_scope = "explicit_actions"
    else:
        expected_actions = primitive_protocol_actions
        library_scope = "primitive_protocol"

    profile = require_analysis_profile(str(metadata["profile_id"]))
    channel_ids = np.asarray(source_arrays["channel_ids"])
    ordered_channel_schema = [
        {
            "sensor_id": int(channel.sensor_id),
            "side": channel.side,
            "muscle_slug": channel.muscle_slug,
        }
        for channel in profile.channels
    ]
    ordered_channel_schema_sha256 = _canonical_json_sha256(ordered_channel_schema)

    scale = action_balanced_channel_scale(
        recorded_values,
        action_columns,
        config.channel_normalization,
    )
    balanced = recorded_values / scale[:, None]
    column_weights, action_multipliers = action_balance_weights(
        recorded_values.shape[1],
        action_columns,
    )
    fit_matrix = balanced * column_weights[None, :]
    diagnostics = channel_scale_diagnostics(recorded_values, scale)
    repo_root = Path(__file__).resolve().parents[2]
    qc_review = _load_qc_review(
        qc_review_manifest,
        source_content_sha256=source_integrity["content_sha256"],
        diagnostics=diagnostics,
        repo_root=repo_root,
    )
    source_provenance = _source_code_provenance()

    scan_rows, selected_k, selection_method = _scan_k(
        balanced,
        fit_matrix,
        boundaries,
        trials,
        action_trial_indices,
        config,
    )
    selected_row = next(row for row in scan_rows if row["k"] == selected_k)
    basis, _, selected_fit_metrics = fit_nmf_best(
        fit_matrix,
        selected_k,
        config.n_init,
        config.seed + selected_k * 1009,
    )
    recorded_basis = scale[:, None] * basis
    time_normalize_samples = metadata.get("time_normalize_samples")
    projected_coefficients, activation_statistics, activation_arrays = _activation_statistics(
        basis,
        balanced,
        trials,
        boundaries,
        action_trial_indices,
        int(time_normalize_samples) if time_normalize_samples is not None else None,
    )
    balanced_vaf, balanced_local_vaf, _ = vaf_metrics(
        balanced,
        basis @ projected_coefficients,
    )
    recorded_vaf, recorded_local_vaf, _ = vaf_metrics(
        recorded_values,
        recorded_basis @ projected_coefficients,
    )
    readiness = _readiness_report(
        metadata,
        actions,
        action_trial_indices,
        expected_actions,
        selected_row,
        diagnostics,
        qc_review,
        source_provenance,
        config,
    )

    basis_hash_arrays = {
        "W": basis.astype(np.float32),
        "W_recorded_units": recorded_basis.astype(np.float32),
        "channel_scale": scale.astype(np.float64),
        "channel_ids": channel_ids,
        "muscle_slugs": source_arrays["muscle_slugs"],
        "sides": source_arrays["sides"],
        "profile_id": np.asarray(metadata["profile_id"]),
        "profile_version": np.asarray(metadata["profile_version"], dtype=np.int16),
        "ordered_channel_schema_sha256": np.asarray(ordered_channel_schema_sha256),
        "k": np.asarray(selected_k, dtype=np.int16),
    }
    basis_sha256 = _arrays_sha256(basis_hash_arrays)
    output_arrays: dict[str, Any] = {
        **basis_hash_arrays,
        "H_projected": projected_coefficients.astype(np.float32),
        "fit_column_weights": column_weights.astype(np.float64),
        "trial_boundaries": boundaries,
        "fs_hz": source_arrays["fs_hz"],
        "trial_ids": np.asarray([str(trial["trial_id"]) for trial in trials]),
        "trial_action_ids": np.asarray([str(trial["action_id"]) for trial in trials]),
        "legacy_dataset_sha256": np.asarray(str(metadata["dataset_sha256"])),
        "source_content_sha256": np.asarray(source_integrity["content_sha256"]),
        "basis_sha256": np.asarray(basis_sha256),
        "schema_version": np.asarray(PRIMITIVE_LIBRARY_SCHEMA_VERSION),
        **activation_arrays,
    }
    scan_payload = _json_safe(
        {
            "schema_version": PRIMITIVE_LIBRARY_SCHEMA_VERSION,
            "library_id": library_id,
            "selection_method": selection_method,
            "selected_k": selected_k,
            "selected_k_stability_pass": selected_row["stability_pass"],
            "selected_k_adequacy_pass": selected_row["adequacy_pass"],
            "selected_k_selection_pass": selected_row["selection_pass"],
            "thresholds": asdict(config),
            "by_k": scan_rows,
        }
    )
    action_balance = {
        action: {
            "trial_count": len(action_trial_indices[action]),
            "columns": int(action_columns[action].size),
            "fit_column_multiplier": action_multipliers[action],
            "total_squared_fit_weight": float(np.sum(column_weights[action_columns[action]] ** 2)),
        }
        for action in actions
    }

    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        library_path = atomic_save_npz(
            staging_dir / PRIMITIVE_LIBRARY_NPZ_NAME,
            **output_arrays,
        )
        library_npz_sha256 = _file_sha256(library_path)
        scan_path = atomic_write_json(staging_dir / PRIMITIVE_LIBRARY_SCAN_NAME, scan_payload)
        scan_sha256 = _file_sha256(scan_path)
        manifest: dict[str, Any] = {
            "schema_version": PRIMITIVE_LIBRARY_SCHEMA_VERSION,
            "artifact_version": 2,
            "library_id": library_id,
            "created_time": datetime.now(UTC).isoformat(),
            "software_version": __version__,
            "source_code_provenance": source_provenance,
            "release": readiness,
            "observation_space_contract": {
                "observation_space_only": True,
                "source_channel_count": int(recorded_values.shape[0]),
                "training_enabled": False,
                "policy_action_basis": False,
                "reverse_mapping_15_or_16_to_354_allowed": False,
                "actuator_space_dimension": 354,
                "allowed_use": [
                    "measurement-side descriptive analysis",
                    "evaluation after an explicit reviewed 354-to-electrode projection",
                ],
                "prohibited_use": [
                    "lifting measured 15/16-channel W into a 354-actuator action basis",
                    "enabling a PPO reward or policy constraint from this manifest",
                ],
            },
            "library_scope": library_scope,
            "expected_primitive_actions": expected_actions,
            "included_actions": actions,
            "participants": metadata.get("participants", []),
            "sessions": metadata.get("sessions", []),
            "trial_ids": [str(trial["trial_id"]) for trial in trials],
            "source_dataset": {
                "npz": {
                    **_path_identity(dataset_path, repo_root),
                    "sha256": source_integrity["npz_file_sha256"],
                },
                "json": {
                    **_path_identity(dataset_path.with_suffix(".json"), repo_root),
                    "sha256": source_integrity["json_file_sha256"],
                },
                **source_integrity,
                "all_semantic_arrays_and_metadata_hash_verified": True,
                "scope": metadata.get("scope"),
                "matrix_shape": list(recorded_values.shape),
            },
            "profile_id": metadata["profile_id"],
            "profile_version": metadata["profile_version"],
            "channel_profile_snapshot": metadata["channel_profile_snapshot"],
            "protocol_id": metadata["protocol_id"],
            "processing": metadata.get("processing"),
            "normalization_references": metadata.get("normalization_references"),
            "crop": {
                "mode": metadata.get("crop_mode"),
                "exploratory": bool(metadata.get("exploratory")),
                "event_semantics": metadata.get("event_semantics"),
                "time_normalize_samples": time_normalize_samples,
            },
            "source_trials": trials,
            "quality": {
                "only_valid": metadata.get("only_valid"),
                "only_valid_includes_preprocessing_qc": metadata.get("only_valid_includes_preprocessing_qc"),
                "channel_scale_diagnostics": diagnostics,
                "channel_diagnostics_sha256": _channel_diagnostics_sha256(diagnostics),
                "independent_qc_review": qc_review,
            },
            "action_balancing": {
                "method": "equal_total_squared_fit_weight_per_action_using_all_columns",
                "channel_scale_distribution": "equal_action_mass",
                "resampling_contract": (
                    "whole-trial action-stratified; equal action squared weight rebuilt "
                    "inside every split and bootstrap draw"
                ),
                "actions": action_balance,
            },
            "basis": {
                "selected_k": selected_k,
                "selection_method": selection_method,
                "selected_k_stability_pass": selected_row["stability_pass"],
                "selected_k_adequacy_pass": selected_row["adequacy_pass"],
                "selected_k_selection_pass": selected_row["selection_pass"],
                "W_space": "action_and_channel_balanced_fit_space_column_l2_normalized",
                "W_recorded_units_relation": "W_recorded_units = channel_scale[:,None] * W",
                "H_semantics": "fixed-W NNLS projection of every unweighted channel-balanced source trial",
                "channel_normalization": config.channel_normalization,
                "channel_scale": scale.tolist(),
                "ordered_channel_schema": ordered_channel_schema,
                "ordered_channel_schema_sha256": ordered_channel_schema_sha256,
                "basis_sha256": basis_sha256,
                "selected_fit_metrics": selected_fit_metrics,
                "balanced_projection_global_vaf": balanced_vaf,
                "balanced_projection_local_vaf": balanced_local_vaf.tolist(),
                "recorded_unit_projection_global_vaf": recorded_vaf,
                "recorded_unit_projection_local_vaf": recorded_local_vaf.tolist(),
                "selected_rank_evidence": selected_row,
                "k_scan": {"path": PRIMITIVE_LIBRARY_SCAN_NAME, "sha256": scan_sha256},
            },
            "action_activation_statistics": activation_statistics,
            "artifacts": {
                "library_npz": PRIMITIVE_LIBRARY_NPZ_NAME,
                "library_npz_sha256": library_npz_sha256,
                "k_scan_json": PRIMITIVE_LIBRARY_SCAN_NAME,
                "k_scan_json_sha256": scan_sha256,
            },
            "manifest_hash_contract": {
                "algorithm": "sha256",
                "payload": "canonical JSON object",
                "canonicalization": "sort_keys UTF-8 separators=(',', ':') allow_nan=false",
                "excluded_fields": ["manifest_sha256"],
            },
        }
        manifest = _json_safe(manifest)
        manifest["manifest_sha256"] = _canonical_json_sha256(manifest)
        atomic_write_json(staging_dir / PRIMITIVE_LIBRARY_MANIFEST_NAME, manifest)
        verify_primitive_synergy_library(staging_dir)
        reservation_path = output_dir.parent / f".primitive-library-{library_id}.publish.lock"
        try:
            reservation_fd = os.open(
                reservation_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise FileExistsError(f"Another publisher is reserving primitive library_id {library_id!r}") from exc
        else:
            os.close(reservation_fd)
        try:
            _assert_unique_library_id(output_dir.parent, library_id, ignore=staging_dir)
            if output_dir.exists():
                raise FileExistsError(f"Primitive library output appeared during publication: {output_dir}")
            os.replace(staging_dir, output_dir)
        finally:
            reservation_path.unlink(missing_ok=True)
        try:
            directory_fd = os.open(output_dir.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return output_dir / PRIMITIVE_LIBRARY_MANIFEST_NAME


__all__ = [
    "PRIMITIVE_LIBRARY_SCHEMA_VERSION",
    "QC_REVIEW_SCHEMA_VERSION",
    "SOURCE_DATASET_DIGEST_VERSION",
    "PrimitiveLibraryConfig",
    "action_balance_weights",
    "action_balanced_channel_scale",
    "build_primitive_synergy_library",
    "verify_primitive_synergy_library",
]
