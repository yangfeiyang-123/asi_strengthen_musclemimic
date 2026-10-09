"""Compare a measured sEMG synergy basis with a simulated one.

The two sides are recorded in different spaces.  Jidian fits 16 surface
electrodes; the policy acts on 354 muscle actuators.  The only space in which
they can be compared is the 15-channel observation space defined by
``configs/physiology/emg_badminton_synergy_16_v2_myofullbody_observation_v1.json``,
which states, for each electrode, the actuators that electrode plausibly
observed.  That mapping is a projection (354 -> 15) and is used here only in
that direction: this module never lifts a measured basis into actuator space,
because the weights say what the electrode saw, not that those actuators are
driven together.

Both sides are divided by their own per-channel scale before the geometry is
compared.  Surface EMG in %MVC and simulated activation in [0, 1] carry
unrelated per-channel gains, so without this the cosine between two synergy
vectors would largely report the gain difference between the two systems.
Because the same rule is applied to both, the comparison stays a statement
about which muscles vary together.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.evaluation.emg_eval import (
    map_simulation_activation,
    match_synergy_bases,
    validate_emg_mapping,
)
from musclemimic.synergy.metrics import global_vaf, local_vaf
from musclemimic.synergy.nmf import fit_best_initialization

MEASURED_BASIS_SCHEMA_VERSION = "emg_measured_synergy_basis_v1"
COMPARISON_SCHEMA_VERSION = "emg_measured_simulation_synergy_comparison_v1"

CHANNEL_NORMALIZATIONS = ("none", "unit_max", "unit_variance")
EXCLUDED_MAPPING_STATUS = "excluded_no_verified_model_homolog"


def _channel_scale(matrix: np.ndarray, mode: str) -> np.ndarray:
    """Per-channel divisor for a ``[sample, channel]`` matrix."""

    if mode not in CHANNEL_NORMALIZATIONS:
        raise ValueError(f"channel_normalization must be one of {CHANNEL_NORMALIZATIONS}, got {mode!r}")
    if mode == "none":
        return np.ones(matrix.shape[1], dtype=np.float64)
    scale = matrix.max(axis=0) if mode == "unit_max" else matrix.std(axis=0)
    silent = scale <= np.finfo(float).eps
    if np.any(silent):
        raise ValueError(
            f"Channels {np.flatnonzero(silent).tolist()} are numerically silent and cannot be rescaled by {mode!r}"
        )
    return np.asarray(scale, dtype=np.float64)


def comparable_channel_names(mapping: Mapping[str, Any]) -> list[str]:
    """The EMG channels the mapping considers comparable, in mapping order."""

    contract = validate_emg_mapping(mapping, allow_provisional_mapping=True)
    return [
        str(entry["emg_channel"])
        for entry in contract["channels"]
        if entry.get("mapping_status") != EXCLUDED_MAPPING_STATUS
    ]


def load_measured_synergy_basis(
    artifact_dir: Path,
    *,
    mapping: Mapping[str, Any],
    required_channel_normalization: str | None = "unit_variance",
    allow_fallback_rank: bool = False,
) -> dict[str, Any]:
    """Read a Jidian ``extract-synergy`` artifact into the comparable space.

    The artifact is bound to the mapping by profile id and by exact channel
    order, and the electrodes the mapping excludes are dropped.  The returned
    basis is the one fitted in the balanced space, because that is the side of
    the comparison whose per-channel scaling is already known.
    """

    artifact_dir = Path(artifact_dir)
    metadata = load_json_strict(artifact_dir / "synergy_metadata.json")
    contract = validate_emg_mapping(mapping, allow_provisional_mapping=True)
    binding = mapping["profile_binding"]

    if metadata["profile_id"] != binding["profile_id"]:
        raise ValueError(
            f"Measured basis profile {metadata['profile_id']!r} does not match the mapping's "
            f"profile {binding['profile_id']!r}"
        )
    if int(metadata["profile_version"]) != int(binding["profile_version"]):
        raise ValueError("Measured basis profile version does not match the mapping's profile version")

    # The comparison is run at the measured basis's rank, so an artifact whose K
    # was not chosen but fell back to the largest tried would set the rank for
    # both sides without anything saying so.
    if not allow_fallback_rank:
        if metadata.get("selection_is_fallback_largest_k"):
            raise ValueError(
                f"{artifact_dir} records selection_is_fallback_largest_k=true: its K is the largest tried, "
                "not a selected rank, and comparing at it would impose an unsupported rank on the simulated "
                "side too. Refit with k-min == k-max at a rank the stability evidence supports, or pass "
                "allow_fallback_rank=True to compare anyway."
            )
        # Reconstruction thresholds can also positively select a K that does not
        # survive resampling, which the fallback flag alone would not catch.
        if metadata.get("selected_k_is_reproducible") is False:
            raise ValueError(
                f"{artifact_dir} records selected_k_is_reproducible=false: its K reconstructs the recording "
                "but its synergies do not reappear when the trials are resampled, so comparing at that rank "
                "would carry the instability into the simulated side. Refit at a rank the stability "
                "evidence supports, or pass allow_fallback_rank=True to compare anyway."
            )

    fitted_normalization = metadata["nmf_parameters"].get("channel_normalization")
    if required_channel_normalization is not None and fitted_normalization != required_channel_normalization:
        raise ValueError(
            f"Measured basis was fitted with channel_normalization={fitted_normalization!r} but the "
            f"comparison requires {required_channel_normalization!r}; refit rather than rescale, "
            "because the basis depends on which space the fit balanced"
        )

    with np.load(artifact_dir / "synergy_basis.npz", allow_pickle=False) as payload:
        basis = np.asarray(payload["W"], dtype=np.float64)
        sensor_ids = [int(value) for value in payload["channel_ids"]]
        sides = [str(value) for value in payload["sides"]]
        slugs = [str(value) for value in payload["muscle_slugs"]]

    expected = [int(entry["sensor_id"]) for entry in contract["channels"]]
    if sensor_ids != expected:
        raise ValueError("Measured basis channel order does not match the mapping's sensor order")
    if basis.shape[0] != len(expected):
        raise ValueError("Measured basis row count does not match the mapping's channel count")

    keep: list[int] = []
    excluded: list[dict[str, Any]] = []
    for index, entry in enumerate(contract["channels"]):
        if entry.get("mapping_status") != EXCLUDED_MAPPING_STATUS:
            keep.append(index)
            continue
        excluded.append(
            {
                "sensor_id": int(entry["sensor_id"]),
                # Named from the artifact rather than the mapping so the record
                # says which measured channel was dropped, not which one the
                # mapping expected to drop.
                "emg_channel": f"S{sensor_ids[index]} {sides[index]}:{slugs[index]}",
                "reason": entry.get("exclusion_reason"),
            }
        )
    if not keep:
        raise ValueError("The mapping excludes every channel; there is nothing to compare")
    comparable = basis[keep, :]
    if not np.all(np.isfinite(comparable)) or comparable.min() < -1e-10:
        raise ValueError("Measured basis must be finite and non-negative")
    return {
        "schema_version": MEASURED_BASIS_SCHEMA_VERSION,
        "basis": np.maximum(comparable, 0.0),
        "channel_names": [str(contract["channels"][index]["emg_channel"]) for index in keep],
        "rank": int(comparable.shape[1]),
        "channel_normalization": fitted_normalization,
        "excluded_channels": excluded,
        "source_artifact": str(artifact_dir),
        "dataset_sha256": metadata.get("dataset_sha256"),
        "measured_global_vaf": metadata.get("global_vaf"),
        "measured_recorded_unit_global_vaf": metadata.get("recorded_unit_global_vaf"),
        "profile_id": metadata["profile_id"],
        "actions": metadata.get("actions"),
        "participants": metadata.get("participants"),
    }


def project_activation_to_comparable_channels(
    muscle_activation: np.ndarray,
    *,
    actuator_names: Sequence[str],
    mapping: Mapping[str, Any],
    allow_provisional_mapping: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Project ``[trial, time, 354]`` activation onto the comparable channels.

    Returns a flat ``[sample, channel]`` matrix, which is the layout the
    project's NMF expects, together with the channel names.
    """

    projected, channel_names = map_simulation_activation(
        muscle_activation,
        actuator_names=actuator_names,
        mapping=mapping,
        allow_provisional_mapping=allow_provisional_mapping,
    )
    flattened = projected.reshape(-1, projected.shape[2])
    return np.maximum(np.asarray(flattened, dtype=np.float64), 0.0), list(channel_names)


def compare_measured_and_simulated_synergies(
    measured: Mapping[str, Any],
    simulated_channel_matrix: np.ndarray,
    *,
    simulated_channel_names: Sequence[str],
    channel_normalization: str = "unit_variance",
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Fit the simulated side at the measured rank and match the two bases."""

    measured_basis = np.asarray(measured["basis"], dtype=np.float64)
    rank = int(measured["rank"])
    # The comparison only means "which muscles covary" if both sides were
    # balanced the same way; otherwise the cosine absorbs the difference between
    # the two rules.
    if measured.get("channel_normalization") != channel_normalization:
        raise ValueError(
            f"Measured basis was balanced with {measured.get('channel_normalization')!r} but the "
            f"simulated side would use {channel_normalization!r}. Comparing across two rules makes the "
            "cosine reflect the rules rather than the coordination; use the same one on both sides."
        )
    if list(simulated_channel_names) != list(measured["channel_names"]):
        raise ValueError("Simulated and measured channel names differ; both must use the mapping's order")
    matrix = np.asarray(simulated_channel_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != measured_basis.shape[0]:
        raise ValueError("Simulated matrix must be [sample, channel] with the measured channel count")
    if matrix.shape[0] < rank:
        raise ValueError("Simulated matrix has fewer samples than the measured rank")

    scale = _channel_scale(matrix, channel_normalization)
    balanced = matrix / scale[None, :]
    best, _ = fit_best_initialization(balanced, rank=rank, seeds=list(seeds), max_iter=max_iter, tol=tol)
    simulated_basis = np.asarray(best.basis, dtype=np.float64)
    local = local_vaf(balanced, best.reconstruction)
    similarity = match_synergy_bases(measured_basis, simulated_basis)
    matched = np.asarray(similarity["weight_cosine_similarity"], dtype=np.float64)
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "rank": rank,
        "channel_names": list(measured["channel_names"]),
        "channel_normalization": channel_normalization,
        "measured_channel_normalization": measured.get("channel_normalization"),
        "measured_source_artifact": measured.get("source_artifact"),
        "measured_dataset_sha256": measured.get("dataset_sha256"),
        "simulated_sample_count": int(matrix.shape[0]),
        "simulated_channel_scale": scale.tolist(),
        "simulated_global_vaf": float(global_vaf(balanced, best.reconstruction)),
        "simulated_per_channel_vaf": {
            name: None if not np.isfinite(local[index]) else float(local[index])
            for index, name in enumerate(measured["channel_names"])
        },
        "weight_cosine_similarity": similarity["weight_cosine_similarity"],
        "mean_weight_cosine_similarity": similarity["mean_weight_cosine_similarity"],
        "minimum_weight_cosine_similarity": float(matched.min()),
        "measured_to_simulated_assignment": similarity["first_to_second_assignment"],
        "n_synergies_matched_ge_0_80": int(np.count_nonzero(matched >= 0.80)),
        "comparison_note": (
            "Both sides were divided by their own per-channel scale before the geometry was "
            "compared, so the cosine reflects which muscles covary rather than the gain "
            "difference between surface EMG and simulated activation."
        ),
        "interpretation_scope": (
            "Descriptive agreement in the 15-channel observation space. The mapping is a "
            "projection from actuators to electrodes and is not evidence that a measured "
            "synergy can be imposed on the 354-actuator policy."
        ),
    }
