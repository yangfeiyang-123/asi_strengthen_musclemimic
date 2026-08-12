from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt

from . import __version__
from .models import CHANNEL_NORMALIZATIONS
from .profiles import require_analysis_profile
from .storage import atomic_save_npz, atomic_write_json, git_commit_hash, read_json

__all__ = [
    "CHANNEL_NORMALIZATIONS",
    "channel_scale",
    "extract_synergy",
    "fit_nmf_best",
    "match_synergies",
    "trial_split_half",
    "vaf_metrics",
]


def vaf_metrics(V: np.ndarray, reconstruction: np.ndarray) -> tuple[float, np.ndarray, float]:
    residual = np.asarray(V) - np.asarray(reconstruction)
    sse = float(np.sum(residual**2))
    denominator = float(np.sum(np.asarray(V) ** 2))
    global_vaf = 1.0 - sse / denominator if denominator > 0 else 0.0
    channel_denominator = np.sum(np.asarray(V) ** 2, axis=1)
    local = 1.0 - np.sum(residual**2, axis=1) / np.maximum(channel_denominator, np.finfo(float).eps)
    return float(global_vaf), local, sse


def channel_scale(V: np.ndarray, mode: str) -> np.ndarray:
    """Per-channel divisor applied before NMF.

    NMF minimises a plain sum of squares, so without this the fit is driven by
    whichever electrodes happen to carry the largest amplitude and it spends
    components reproducing them one muscle at a time instead of recovering
    co-activation structure.  Equalising the channels first is the standard
    remedy in the surface-EMG synergy literature.  The basis is converted back
    to the recorded units afterwards, so the scaling changes only what the
    objective weights, never the units the basis is reported in.
    """
    if mode not in CHANNEL_NORMALIZATIONS:
        raise ValueError(f"channel_normalization must be one of {CHANNEL_NORMALIZATIONS}, got {mode!r}")
    matrix = np.asarray(V, dtype=np.float64)
    if mode == "none":
        scale = np.ones(matrix.shape[0], dtype=np.float64)
    elif mode == "unit_max":
        scale = matrix.max(axis=1)
    else:
        scale = matrix.std(axis=1)
    if not np.all(np.isfinite(scale)):
        raise ValueError("Channel scaling produced a non-finite divisor")
    silent = scale <= np.finfo(float).eps
    if np.any(silent) and mode != "none":
        raise ValueError(
            "Channels "
            f"{np.flatnonzero(silent).tolist()} are numerically silent and cannot be "
            f"rescaled by '{mode}'; drop them or use channel_normalization='none'"
        )
    return np.maximum(scale, np.finfo(float).eps)


def channel_scale_diagnostics(
    V: np.ndarray,
    scale: np.ndarray,
    *,
    quiet_ratio_threshold: float = 0.25,
    outlier_peak_threshold: float = 3.0,
) -> dict[str, Any]:
    """Flag channels whose divisor is untrustworthy, without changing the fit.

    Balancing divides every channel up to the same working range, which is what
    makes the fit describe all of them -- but it does that whether the channel
    carries muscle activity or a failing electrode's noise floor, and it cannot
    tell those apart.  Two divisors deserve a second look: one far below the
    others, which lifts a near-silent channel to full weight, and one set by a
    lone peak, which shrinks that channel's typical level and costs it
    influence.  Both are electrode or normalisation problems for the analyst to
    resolve, so they are reported rather than silently corrected here.
    """
    matrix = np.asarray(V, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    median_scale = float(np.median(scale))
    ratio_to_median = scale / max(median_scale, np.finfo(float).eps)
    peak = matrix.max(axis=1)
    typical = np.percentile(matrix, 95, axis=1)
    peak_to_typical = peak / np.maximum(typical, np.finfo(float).eps)
    return {
        "median_scale": median_scale,
        "scale_ratio_to_median": ratio_to_median.tolist(),
        "peak_to_p95_ratio": peak_to_typical.tolist(),
        "quiet_ratio_threshold": quiet_ratio_threshold,
        "outlier_peak_threshold": outlier_peak_threshold,
        "channels_far_below_median_scale": np.flatnonzero(ratio_to_median < quiet_ratio_threshold).tolist(),
        "channels_with_outlier_driven_peak": np.flatnonzero(peak_to_typical > outlier_peak_threshold).tolist(),
    }


def fit_nmf_best(V: np.ndarray, k: int, n_init: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    V = np.maximum(np.asarray(V, dtype=np.float64), 0.0)
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for init_index in range(n_init):
        model = NMF(
            n_components=k,
            init="random",
            random_state=seed + init_index,
            max_iter=2000,
            tol=1e-4,
            solver="cd",
        )
        W = model.fit_transform(V)
        H = model.components_
        error = float(np.sum((V - W @ H) ** 2))
        if best is None or error < best[2]:
            best = (W, H, error)
    assert best is not None
    W, H, _ = best
    norms = np.linalg.norm(W, axis=0)
    norms = np.maximum(norms, np.finfo(float).eps)
    W = W / norms
    H = H * norms[:, None]
    global_vaf, local_vaf, sse = vaf_metrics(V, W @ H)
    return W, H, {
        "k": k,
        "global_vaf": global_vaf,
        "local_vaf": local_vaf.tolist(),
        "local_vaf_fraction_ge_0_75": float(np.mean(local_vaf >= 0.75)),
        "reconstruction_sse": sse,
        "frobenius_error": float(np.sqrt(sse)),
    }


def match_synergies(reference_W: np.ndarray, candidate_W: np.ndarray) -> dict[str, Any]:
    ref = reference_W / np.maximum(np.linalg.norm(reference_W, axis=0, keepdims=True), np.finfo(float).eps)
    candidate = candidate_W / np.maximum(np.linalg.norm(candidate_W, axis=0, keepdims=True), np.finfo(float).eps)
    similarity = ref.T @ candidate
    rows, cols = linear_sum_assignment(-similarity)
    values = similarity[rows, cols]
    return {
        "reference_indices": rows.tolist(),
        "candidate_indices": cols.tolist(),
        "cosine_similarities": values.tolist(),
        "mean_cosine_similarity": float(np.mean(values)),
        "minimum_cosine_similarity": float(np.min(values)),
    }


def trial_split_half(
    V: np.ndarray,
    k: int,
    n_init: int,
    seed: int,
    boundaries: np.ndarray,
    repeats: int,
) -> dict[str, Any]:
    """Split-half agreement over repeated random partitions of whole trials.

    Splitting on trials rather than on the raw time axis keeps each half a set
    of complete movements; a mid-time cut would put the first part of a stroke
    in one half and its continuation in the other and report agreement that
    reflects the cut, not the reproducibility of the synergies.

    ``n_init`` is used as given.  Fitting each half with fewer restarts than the
    basis being checked leaves the halves in worse local minima and reports
    disagreement that came from the optimiser rather than from the data, so the
    cost/precision tradeoff belongs to the caller.
    """
    if repeats < 1:
        raise ValueError("split-half requires repeats >= 1")
    trial_count = len(boundaries) - 1
    if trial_count < 4:
        return {"available": False, "reason": "requires at least 4 trials"}
    slices = [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])]
    generator = np.random.default_rng(seed)
    runs: list[dict[str, Any]] = []
    for repeat in range(repeats):
        order = generator.permutation(trial_count)
        half = trial_count // 2
        groups = (order[:half], order[half:])
        columns = [np.concatenate([np.arange(*slices[i]) for i in group]) for group in groups]
        if any(len(column) < k for column in columns):
            continue
        W_a, _, _ = fit_nmf_best(V[:, columns[0]], k, n_init, seed + 10000 + repeat * 7)
        W_b, _, _ = fit_nmf_best(V[:, columns[1]], k, n_init, seed + 20000 + repeat * 7)
        match = match_synergies(W_a, W_b)
        match["repeat_index"] = repeat
        match["trial_counts"] = [len(group) for group in groups]
        runs.append(match)
    if not runs:
        return {"available": False, "reason": "insufficient samples per half"}
    per_run_min = [item["minimum_cosine_similarity"] for item in runs]
    per_run_mean = [item["mean_cosine_similarity"] for item in runs]
    return {
        "available": True,
        "repeats": len(runs),
        # A partition whose half was smaller than k is skipped, and those are
        # exactly the uneven splits, so the surviving runs are biased toward
        # larger halves.  Report the shortfall instead of only the survivors.
        "requested_repeats": repeats,
        "skipped_repeats": repeats - len(runs),
        "partition": "random halves of whole trials",
        "mean_cosine_similarity": float(np.mean(per_run_mean)),
        "mean_minimum_cosine_similarity": float(np.mean(per_run_min)),
        "worst_minimum_cosine_similarity": float(np.min(per_run_min)),
        "fraction_of_repeats_all_synergies_ge_0_80": float(np.mean(np.asarray(per_run_min) >= 0.80)),
        "runs": runs,
    }


def _stability(
    V: np.ndarray,
    reference_W: np.ndarray,
    k: int,
    n_init: int,
    seed: int,
    boundaries: np.ndarray,
    split_half_repeats: int = 20,
) -> dict[str, Any]:
    result: dict[str, Any] = {"matching_method": "Hungarian assignment on cosine similarity"}
    # Each half is a smaller problem than the full fit, so it needs fewer
    # restarts to reach the same quality; the cap keeps the repeated splits
    # affordable without letting the optimiser dominate the agreement.
    result["split_half"] = trial_split_half(
        V, k, max(2, min(n_init, 25)), seed, boundaries, split_half_repeats
    )
    trial_matches = []
    for trial_index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        segment = V[:, int(start):int(stop)]
        if segment.shape[1] < k:
            continue
        trial_W, _, _ = fit_nmf_best(segment, k, max(2, min(n_init, 5)), seed + 30000 + trial_index * 100)
        match = match_synergies(reference_W, trial_W)
        match["trial_index"] = trial_index
        trial_matches.append(match)
    result["cross_trial"] = {
        "matches": trial_matches,
        "mean_cosine_similarity": float(np.mean([item["mean_cosine_similarity"] for item in trial_matches]))
        if trial_matches else None,
    }
    return result


def _save_plots(output_dir: Path, metrics_by_k: list[dict[str, Any]], W: np.ndarray, labels: list[str]) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    ks = [item["k"] for item in metrics_by_k]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(ks, [item["global_vaf"] for item in metrics_by_k], marker="o", label="Global VAF (fit space)")
    if all("recorded_unit_global_vaf" in item for item in metrics_by_k):
        axis.plot(
            ks,
            [item["recorded_unit_global_vaf"] for item in metrics_by_k],
            marker="s",
            linestyle=":",
            label="Global VAF (recorded units)",
        )
    axis.plot(
        ks,
        [np.min(item["local_vaf"]) for item in metrics_by_k],
        marker="^",
        linestyle="--",
        label="Worst muscle VAF",
    )
    axis.axhline(0.90, color="tab:red", linestyle="--", linewidth=1, label="0.90 threshold")
    axis.set(xlabel="K", ylabel="VAF", ylim=(0, 1.02), title="NMF model selection")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "vaf_by_k.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(max(6, W.shape[1] * 1.2), 8))
    image = axis.imshow(W, aspect="auto", cmap="viridis")
    axis.set_xticks(range(W.shape[1]), [f"S{k + 1}" for k in range(W.shape[1])])
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("L2-normalized synergy weights W")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(plots / "synergy_weights.png", dpi=180)
    plt.close(fig)


def extract_synergy(
    dataset_path: Path,
    output_dir: Path,
    k_min: int = 1,
    k_max: int = 8,
    n_init: int = 50,
    seed: int = 20260720,
    global_vaf_threshold: float = 0.90,
    local_vaf_threshold: float = 0.75,
    local_fraction_required: float = 0.8,
    include_h: bool = True,
    calculate_stability: bool = True,
    channel_normalization: str = "unit_variance",
    split_half_repeats: int = 20,
    reproducible_fraction_required: float = 0.5,
) -> Path:
    if k_min < 1 or k_max < k_min or n_init < 1:
        raise ValueError("Require 1 <= k_min <= k_max and n_init >= 1")
    with np.load(dataset_path, allow_pickle=False) as payload:
        V = np.asarray(payload["V"], dtype=np.float64)
        channel_ids = payload["channel_ids"]
        muscle_slugs = payload["muscle_slugs"]
        sides = payload["sides"]
        boundaries = payload["trial_boundaries"]
        fs_hz = float(payload["fs_hz"])
    if V.ndim != 2 or np.any(~np.isfinite(V)) or np.any(V < 0):
        raise ValueError("NMF input V must be finite, nonnegative, and shaped [channels, time]")
    dataset_metadata = read_json(dataset_path.with_suffix(".json"))
    require_analysis_profile(str(dataset_metadata["profile_id"]))
    signature = tuple(int(value) for value in channel_ids)
    expected = tuple(channel["sensor_id"] for channel in dataset_metadata["channel_profile_snapshot"]["channels"])
    if signature != expected:
        raise ValueError("Dataset channel order does not match its channel profile snapshot")
    scale = channel_scale(V, channel_normalization)
    V_fit = V / scale[:, None]
    labels = [f"S{sensor} {side}:{slug}" for sensor, side, slug in zip(channel_ids, sides, muscle_slugs)]
    scale_diagnostics = channel_scale_diagnostics(V, scale)
    for index in scale_diagnostics["channels_far_below_median_scale"]:
        print(
            f"WARNING: {labels[index]} has a channel scale "
            f"{scale_diagnostics['scale_ratio_to_median'][index]:.2f}x the median. Balancing divides it up "
            "to the same working range as the rest, which it does whether the channel carries muscle "
            "activity or a failing electrode's noise floor. Check the electrode before reading its weight."
        )
    if channel_normalization == "unit_max":
        # Only 'unit_max' takes its divisor from the extreme value, so only there
        # does a lone peak shrink the channel's typical level and cost it
        # influence.  The ratio stays in the metadata for every mode.
        for index in scale_diagnostics["channels_with_outlier_driven_peak"]:
            print(
                f"WARNING: {labels[index]} has a peak {scale_diagnostics['peak_to_p95_ratio'][index]:.1f}x its "
                "95th percentile, so its 'unit_max' divisor is set by a few samples, which shrinks the "
                "channel's typical level and costs it influence in the fit."
            )
    metrics_by_k: list[dict[str, Any]] = []
    fits: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    max_components = min(k_max, V.shape[0], V.shape[1])
    for k in range(k_min, max_components + 1):
        W, H, metrics = fit_nmf_best(V_fit, k, n_init, seed + k * 1000)
        fits[k] = (W, H)
        recorded_vaf, recorded_local, _ = vaf_metrics(V, (W * scale[:, None]) @ H)
        metrics["recorded_unit_global_vaf"] = recorded_vaf
        metrics["recorded_unit_local_vaf"] = recorded_local.tolist()
        metrics_by_k.append(metrics)
        print(f"K={k}: global VAF={metrics['global_vaf']:.4f}, local>=.75={metrics['local_vaf_fraction_ge_0_75']:.2f}")
    qualifying = [
        item["k"]
        for item in metrics_by_k
        if item["global_vaf"] >= global_vaf_threshold
        and np.mean(np.asarray(item["local_vaf"]) >= local_vaf_threshold) >= local_fraction_required
    ]
    # With a single candidate the caller pinned K, so neither "met the
    # thresholds" nor "fell back to the largest tried" describes what happened.
    pinned_by_caller = len(metrics_by_k) == 1
    selection_is_fallback = None if pinned_by_caller else not qualifying
    selected = qualifying[0] if qualifying else metrics_by_k[-1]["k"]
    if selection_is_fallback:
        # Falling back to the largest K tried is not a selection, and the VAF
        # thresholds were calibrated on unbalanced fits where the loudest
        # channels inflate them.  Say so rather than let the largest K be read
        # as the number of synergies the data supports.
        print(
            f"WARNING: no K in [{k_min}, {max_components}] met global VAF >= {global_vaf_threshold} with "
            f">= {local_fraction_required:.0%} of muscles at local VAF >= {local_vaf_threshold}. "
            f"Falling back to K={selected}, the largest tried. Treat it as unselected and choose K from "
            f"the split-half stability in metrics.json, not from this value."
        )
    W, H = fits[selected]
    selected_metrics = next(item for item in metrics_by_k if item["k"] == selected)
    stability = (
        _stability(V_fit, W, selected, n_init, seed, boundaries, split_half_repeats)
        if calculate_stability
        else {"calculated": False}
    )
    # Reconstruction thresholds can positively select a K whose synergies do not
    # survive resampling -- VAF keeps rising as components split muscle by
    # muscle, and a threshold on it rewards exactly that.  So the selected K is
    # checked against its own split-half agreement, not only against the
    # fallback case.
    split_half = stability.get("split_half", {}) if isinstance(stability, dict) else {}
    reproducible_fraction = split_half.get("fraction_of_repeats_all_synergies_ge_0_80")
    selected_k_is_reproducible: bool | None = (
        None if reproducible_fraction is None else bool(reproducible_fraction >= reproducible_fraction_required)
    )
    if selected_k_is_reproducible is False:
        print(
            f"WARNING: at K={selected} only {reproducible_fraction:.0%} of split-half partitions reproduced "
            f"every synergy at cosine >= 0.80 (required {reproducible_fraction_required:.0%}). This K "
            "reconstructs the recording but does not describe a coordination pattern that reappears when "
            "the trials are resampled. Choose K from the stability evidence before quoting a synergy count."
        )
    W_recorded = W * scale[:, None]
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "W": W.astype(np.float32),
        "W_recorded_units": W_recorded.astype(np.float32),
        "channel_scale": scale.astype(np.float64),
        "channel_ids": channel_ids,
        "muscle_slugs": muscle_slugs,
        "sides": sides,
        "k": np.asarray(selected, dtype=np.int16),
        "fs_hz": np.asarray(fs_hz),
    }
    if include_h:
        arrays["H"] = H.astype(np.float32)
    basis_path = atomic_save_npz(output_dir / "synergy_basis.npz", **arrays)
    metrics = {
        "k_search": metrics_by_k,
        "selected_k": selected,
        "selection_is_fallback_largest_k": selection_is_fallback,
        "selected_k_is_reproducible": selected_k_is_reproducible,
        "reproducible_fraction_required": reproducible_fraction_required,
        "selection_thresholds": {
            "global_vaf": global_vaf_threshold,
            "local_vaf": local_vaf_threshold,
            "local_fraction_required": local_fraction_required,
        },
        "selected": selected_metrics,
        "stability": stability,
    }
    atomic_write_json(output_dir / "metrics.json", metrics)
    source_types = sorted({item["category"] for item in dataset_metadata["trials"]})
    metadata = {
        "artifact_version": 2,
        "software_version": __version__,
        "profile_id": dataset_metadata["profile_id"],
        "profile_version": dataset_metadata["profile_version"],
        "channel_profile_snapshot": dataset_metadata["channel_profile_snapshot"],
        "participants": dataset_metadata["participants"],
        "sessions": dataset_metadata["sessions"],
        "trials": [item["trial_id"] for item in dataset_metadata["trials"]],
        "actions": dataset_metadata["actions"],
        "source_categories": source_types,
        "normalization": dataset_metadata["processing"]["normalization_method"],
        "processing": dataset_metadata["processing"],
        "crop_mode": dataset_metadata["crop_mode"],
        "crop_is_exploratory": dataset_metadata["exploratory"],
        # Carried through so a reader of H knows whether its columns are a
        # movement percentage or wall-clock samples without reopening the dataset.
        "time_normalize_samples": dataset_metadata["time_normalize_samples"],
        "trial_boundaries": dataset_metadata["trial_boundaries"],
        "nmf_parameters": {
            "k_min": k_min,
            "k_max": k_max,
            "n_init": n_init,
            "random_seed": seed,
            "W_normalization": "column_l2_unit_norm_with_inverse_scale_applied_to_H",
            "permutation_comparison": "Hungarian assignment on cosine similarity",
            "channel_normalization": channel_normalization,
            "channel_scale_recorded_units": scale.tolist(),
            "channel_scale_diagnostics": scale_diagnostics,
            "fit_space_note": (
                "V was divided by channel_scale before NMF so that every electrode contributes "
                "comparably to the least-squares objective. 'W' is the basis in that balanced "
                "space; 'W_recorded_units' = channel_scale[:, None] * W restores the basis to "
                "the recorded normalization units, and both reconstruct with the same H."
            ),
            "split_half_repeats": split_half_repeats,
        },
        "selected_k": selected,
        # Both travel with selected_k so a consumer reading only this file
        # cannot mistake "largest K tried" for "the K the thresholds chose",
        # nor a well-reconstructing K for one that reappears on resampling.
        "selection_is_fallback_largest_k": selection_is_fallback,
        "selected_k_is_reproducible": selected_k_is_reproducible,
        "global_vaf": selected_metrics["global_vaf"],
        "local_vaf": selected_metrics["local_vaf"],
        "recorded_unit_global_vaf": selected_metrics["recorded_unit_global_vaf"],
        "recorded_unit_local_vaf": selected_metrics["recorded_unit_local_vaf"],
        "stability": stability,
        "dataset_sha256": dataset_metadata["dataset_sha256"],
        "git_commit_hash": git_commit_hash(Path.cwd()),
        "created_time": datetime.now(timezone.utc).isoformat(),
        "model_mapping_note": "W is expressed only in the experimental surface-EMG channel space; no musculoskeletal-model mapping is inferred.",
    }
    atomic_write_json(output_dir / "synergy_metadata.json", metadata)
    _save_plots(output_dir, metrics_by_k, W, labels)
    return basis_path
