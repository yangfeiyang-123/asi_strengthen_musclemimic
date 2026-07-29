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
from .profiles import require_analysis_profile
from .storage import atomic_save_npz, atomic_write_json, git_commit_hash, read_json


def vaf_metrics(V: np.ndarray, reconstruction: np.ndarray) -> tuple[float, np.ndarray, float]:
    residual = np.asarray(V) - np.asarray(reconstruction)
    sse = float(np.sum(residual**2))
    denominator = float(np.sum(np.asarray(V) ** 2))
    global_vaf = 1.0 - sse / denominator if denominator > 0 else 0.0
    channel_denominator = np.sum(np.asarray(V) ** 2, axis=1)
    local = 1.0 - np.sum(residual**2, axis=1) / np.maximum(channel_denominator, np.finfo(float).eps)
    return float(global_vaf), local, sse


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


def _stability(
    V: np.ndarray,
    reference_W: np.ndarray,
    k: int,
    n_init: int,
    seed: int,
    boundaries: np.ndarray,
) -> dict[str, Any]:
    split = V.shape[1] // 2
    result: dict[str, Any] = {"matching_method": "Hungarian assignment on cosine similarity"}
    if split >= k and V.shape[1] - split >= k:
        W_a, _, _ = fit_nmf_best(V[:, :split], k, max(2, min(n_init, 10)), seed + 10000)
        W_b, _, _ = fit_nmf_best(V[:, split:], k, max(2, min(n_init, 10)), seed + 20000)
        result["split_half"] = match_synergies(W_a, W_b)
    else:
        result["split_half"] = {"available": False, "reason": "insufficient samples"}
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
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot([item["k"] for item in metrics_by_k], [item["global_vaf"] for item in metrics_by_k], marker="o", label="Global VAF")
    axis.axhline(0.90, color="tab:red", linestyle="--", label="0.90 threshold")
    axis.set(xlabel="K", ylabel="VAF", ylim=(0, 1.02), title="NMF model selection")
    axis.grid(alpha=0.3)
    axis.legend()
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
    metrics_by_k: list[dict[str, Any]] = []
    fits: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    max_components = min(k_max, V.shape[0], V.shape[1])
    for k in range(k_min, max_components + 1):
        W, H, metrics = fit_nmf_best(V, k, n_init, seed + k * 1000)
        fits[k] = (W, H)
        metrics_by_k.append(metrics)
        print(f"K={k}: global VAF={metrics['global_vaf']:.4f}, local>=.75={metrics['local_vaf_fraction_ge_0_75']:.2f}")
    selected = next(
        (
            item["k"]
            for item in metrics_by_k
            if item["global_vaf"] >= global_vaf_threshold
            and np.mean(np.asarray(item["local_vaf"]) >= local_vaf_threshold) >= local_fraction_required
        ),
        metrics_by_k[-1]["k"],
    )
    W, H = fits[selected]
    selected_metrics = next(item for item in metrics_by_k if item["k"] == selected)
    stability = _stability(V, W, selected, n_init, seed, boundaries) if calculate_stability else {"calculated": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "W": W.astype(np.float32),
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
        "artifact_version": 1,
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
        "nmf_parameters": {
            "k_min": k_min,
            "k_max": k_max,
            "n_init": n_init,
            "random_seed": seed,
            "W_normalization": "column_l2_unit_norm_with_inverse_scale_applied_to_H",
            "permutation_comparison": "Hungarian assignment on cosine similarity",
        },
        "selected_k": selected,
        "global_vaf": selected_metrics["global_vaf"],
        "local_vaf": selected_metrics["local_vaf"],
        "stability": stability,
        "dataset_sha256": dataset_metadata["dataset_sha256"],
        "git_commit_hash": git_commit_hash(Path.cwd()),
        "created_time": datetime.now(timezone.utc).isoformat(),
        "model_mapping_note": "W is expressed only in the experimental surface-EMG channel space; no musculoskeletal-model mapping is inferred.",
    }
    atomic_write_json(output_dir / "synergy_metadata.json", metadata)
    labels = [f"S{sensor} {side}:{muscle}" for sensor, side, muscle in zip(channel_ids, sides, muscle_slugs)]
    _save_plots(output_dir, metrics_by_k, W, labels)
    return basis_path
