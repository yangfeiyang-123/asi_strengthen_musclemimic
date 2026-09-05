"""Does a complete stroke reuse the synergies of the basic movements?

Fitting each action separately answers "what does this action look like" but not
"is this the same control structure".  A basic-movement basis that reconstructs
a complete stroke it never saw is evidence of reuse; one that does not says the
complete stroke needs something new.  Both directions are measured here on
held-out data, because training-set VAF rises with rank whether or not anything
is shared.

The definitions mirror the simulation-side ones in ``musclemimic.synergy`` so
the two halves of the study stay comparable:

- novelty of a candidate synergy is the non-negative-cone (NNLS) residual of
  that column against the reference basis, over the column norm, and a column
  is a duplicate when its cosine to some reference column is at or above the
  duplicate threshold (``musclemimic/synergy/hybrid_basis.py``);
- effective rank is ``exp`` of the entropy of the normalised singular values,
  and the condition number is ``sigma_max / sigma_min``
  (``musclemimic/synergy/action_interface.py``);
- bases are compared by Hungarian assignment on column cosine.

Every matrix here is ``[channels, time]`` and every basis ``[channels, rank]``,
matching the rest of this package.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import nnls

from .synergy import channel_scale, fit_nmf_best, match_synergies, vaf_metrics

# Both mirror musclemimic/synergy/hybrid_basis.py::HybridBasisConfig.
DEFAULT_NOVELTY_RESIDUAL_RATIO = 0.15
DEFAULT_DUPLICATE_COSINE_SIMILARITY = 0.95

_EPSILON = 1e-12


def basis_geometry(W: np.ndarray) -> dict[str, float]:
    """Condition number and entropy-based effective rank of a basis."""
    basis = np.asarray(W, dtype=np.float64)
    singular = np.linalg.svd(basis, compute_uv=False)
    if singular.size == 0 or singular[0] <= 0.0:
        raise ValueError("basis has no effective directions")
    tolerance = np.finfo(np.float64).eps * max(basis.shape) * singular[0]
    positive = singular[singular > tolerance]
    condition = float("inf") if positive.size != min(basis.shape) else float(positive[0] / positive[-1])
    probabilities = singular / np.sum(singular)
    entropy = -float(np.sum(np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0)))
    effective_rank = float(np.exp(entropy))
    return {
        "condition_number": condition,
        "effective_rank": effective_rank,
        "effective_rank_fraction": effective_rank / max(1, basis.shape[1]),
        "singular_values": singular.tolist(),
    }


def initialization_stability(V: np.ndarray, k: int, n_init: int, seed: int, *, restarts: int = 8) -> dict[str, Any]:
    """Agreement between independently seeded fits of the same data.

    This isolates optimiser variance from data variance: if repeated restarts on
    identical data disagree, a disagreement between two halves says nothing
    about the halves.
    """
    if restarts < 2:
        raise ValueError("initialization stability needs at least 2 restarts")
    bases = [fit_nmf_best(V, k, n_init, seed + index * 104729)[0] for index in range(restarts)]
    scores = [
        match_synergies(bases[i], bases[j])["mean_cosine_similarity"]
        for i in range(len(bases))
        for j in range(i + 1, len(bases))
    ]
    return {
        "restarts": restarts,
        "pair_count": len(scores),
        "mean_cosine_similarity": float(np.mean(scores)),
        "minimum_cosine_similarity": float(np.min(scores)),
    }


def bootstrap_stability(
    V: np.ndarray,
    reference_W: np.ndarray,
    k: int,
    n_init: int,
    seed: int,
    boundaries: np.ndarray,
    repeats: int = 20,
) -> dict[str, Any]:
    """Agreement with the reference basis under trials resampled with replacement.

    Resampling whole trials rather than time samples keeps each draw a set of
    complete movements, and makes the spread reflect which repetitions were
    recorded rather than which instants were drawn.
    """
    if repeats < 1:
        raise ValueError("bootstrap stability requires repeats >= 1")
    trial_count = len(boundaries) - 1
    if trial_count < 2:
        return {"available": False, "reason": "requires at least 2 trials"}
    slices = [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])]
    generator = np.random.default_rng(seed)
    scores: list[float] = []
    for repeat in range(repeats):
        drawn = generator.integers(0, trial_count, size=trial_count)
        columns = np.concatenate([np.arange(*slices[index]) for index in drawn])
        if columns.size < k:
            continue
        fitted, _, _ = fit_nmf_best(V[:, columns], k, n_init, seed + 5000 + repeat * 13)
        scores.append(match_synergies(reference_W, fitted)["mean_cosine_similarity"])
    if not scores:
        return {"available": False, "reason": "insufficient samples per resample"}
    return {
        "available": True,
        "requested_repeats": repeats,
        "repeats": len(scores),
        "skipped_repeats": repeats - len(scores),
        "resample": "whole trials with replacement",
        "mean_cosine_similarity": float(np.mean(scores)),
        "minimum_cosine_similarity": float(np.min(scores)),
        "median_cosine_similarity": float(np.median(scores)),
    }


def project_onto_basis(W: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative least-squares projection of ``V`` onto a fixed basis.

    The basis is held fixed and only the coefficients are solved for, which is
    what makes the resulting VAF a held-out measurement rather than a refit.
    """
    basis = np.asarray(W, dtype=np.float64)
    values = np.maximum(np.asarray(V, dtype=np.float64), 0.0)
    if basis.shape[0] != values.shape[0]:
        raise ValueError("basis and data must share the channel axis")
    coefficients = np.empty((basis.shape[1], values.shape[1]), dtype=np.float64)
    for index in range(values.shape[1]):
        coefficients[:, index], _ = nnls(basis, values[:, index])
    return coefficients, basis @ coefficients


def heldout_reconstruction(W_source: np.ndarray, V_target: np.ndarray) -> dict[str, Any]:
    """How much of an unseen action the source synergies already explain."""
    _, reconstruction = project_onto_basis(W_source, V_target)
    global_vaf, local_vaf, _ = vaf_metrics(np.maximum(V_target, 0.0), reconstruction)
    return {
        "heldout_global_vaf": global_vaf,
        "heldout_local_vaf": local_vaf.tolist(),
        "heldout_local_vaf_fraction_ge_0_75": float(np.mean(local_vaf >= 0.75)),
        "worst_channel_heldout_vaf": float(np.min(local_vaf)),
    }


def synergy_novelty(
    reference_W: np.ndarray,
    candidate_W: np.ndarray,
    *,
    novelty_residual_ratio: float = DEFAULT_NOVELTY_RESIDUAL_RATIO,
    duplicate_cosine_similarity: float = DEFAULT_DUPLICATE_COSINE_SIMILARITY,
) -> dict[str, Any]:
    """Which of the candidate's synergies the reference basis cannot express.

    A column counts as new only if the reference cone leaves a large residual
    *and* no single reference column is nearly parallel to it -- the two catch
    different things, since a column can be reachable as a combination while
    matching nothing on its own.
    """
    reference = np.asarray(reference_W, dtype=np.float64)
    candidate = np.asarray(candidate_W, dtype=np.float64)
    reference_norms = np.maximum(np.linalg.norm(reference, axis=0), _EPSILON)
    columns: list[dict[str, Any]] = []
    for index in range(candidate.shape[1]):
        column = candidate[:, index]
        column_norm = float(np.linalg.norm(column))
        if column_norm <= _EPSILON:
            raise ValueError(f"candidate synergy {index} is empty")
        _, residual_norm = nnls(reference, column)
        residual_ratio = float(residual_norm / column_norm)
        cosines = (reference.T @ column) / (reference_norms * column_norm)
        best = int(np.argmax(cosines))
        max_cosine = float(np.clip(cosines[best], -1.0, 1.0))
        is_novel = residual_ratio > novelty_residual_ratio and max_cosine < duplicate_cosine_similarity
        columns.append(
            {
                "candidate_index": index,
                "cone_residual_ratio": residual_ratio,
                "max_reference_cosine": max_cosine,
                "closest_reference_index": best,
                "is_novel": bool(is_novel),
            }
        )
    novel = [item for item in columns if item["is_novel"]]
    return {
        "novelty_residual_ratio_threshold": novelty_residual_ratio,
        "duplicate_cosine_similarity_threshold": duplicate_cosine_similarity,
        "candidate_rank": int(candidate.shape[1]),
        "reference_rank": int(reference.shape[1]),
        "novel_synergy_count": len(novel),
        "novel_synergy_indices": [item["candidate_index"] for item in novel],
        "mean_cone_residual_ratio": float(np.mean([item["cone_residual_ratio"] for item in columns])),
        "columns": columns,
    }


def within_action_heldout(
    V: np.ndarray,
    boundaries: np.ndarray,
    k: int,
    n_init: int,
    seed: int,
    repeats: int = 20,
) -> dict[str, Any]:
    """Held-out VAF from one half of an action's trials to the other half.

    This is the ceiling any cross-action number should be read against.  A basis
    never reaches the VAF of a refit even on the same action -- unseen
    repetitions differ -- so a cross-action held-out VAF is only low if it is low
    *relative to this*, not relative to the training fit.
    """
    if repeats < 1:
        raise ValueError("within-action held-out requires repeats >= 1")
    trial_count = len(boundaries) - 1
    if trial_count < 4:
        return {"available": False, "reason": "requires at least 4 trials"}
    slices = [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])]
    generator = np.random.default_rng(seed)
    scores: list[float] = []
    for repeat in range(repeats):
        order = generator.permutation(trial_count)
        half = trial_count // 2
        fit_columns = np.concatenate([np.arange(*slices[i]) for i in order[:half]])
        test_columns = np.concatenate([np.arange(*slices[i]) for i in order[half:]])
        if fit_columns.size < k or test_columns.size < k:
            continue
        fitted, _, _ = fit_nmf_best(V[:, fit_columns], k, n_init, seed + 7000 + repeat * 11)
        scores.append(heldout_reconstruction(fitted, V[:, test_columns])["heldout_global_vaf"])
    if not scores:
        return {"available": False, "reason": "insufficient samples per half"}
    return {
        "available": True,
        "requested_repeats": repeats,
        "repeats": len(scores),
        "mean_heldout_global_vaf": float(np.mean(scores)),
        "median_heldout_global_vaf": float(np.median(scores)),
        "minimum_heldout_global_vaf": float(np.min(scores)),
    }


def shared_channel_scale(matrices: dict[str, np.ndarray], mode: str) -> np.ndarray:
    """One channel scale for every action in a comparison.

    Bases fitted under different scalings live in different spaces, so a cosine
    between them would partly report the scaling difference.  Sharing one scale
    is what keeps the reuse numbers about coordination.

    Derive it from a fixed reference set -- normally every trial in the session
    -- rather than from whichever actions happen to be in one comparison.
    Otherwise adding or dropping an action rescales the space and every previously
    reported number moves, which makes two comparisons incomparable.
    """
    if not matrices:
        raise ValueError("at least one matrix is required")
    stacked = np.concatenate([np.asarray(value, dtype=np.float64) for value in matrices.values()], axis=1)
    return channel_scale(stacked, mode)


def reuse_analysis(
    matrices: dict[str, np.ndarray],
    boundaries: dict[str, np.ndarray],
    *,
    source_key: str,
    target_keys: list[str],
    rank: int,
    channel_normalization: str = "unit_variance",
    n_init: int = 50,
    seed: int = 20260720,
    target_rank: int | None = None,
    scale: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit synergies on the source action set and test them on unseen actions.

    Both sides are balanced by one shared channel scale, and the source basis is
    held fixed while the target's coefficients are solved, so the reported VAF
    is what the source structure explains rather than what a refit could.
    """
    if source_key not in matrices:
        raise KeyError(f"source {source_key!r} is not among the supplied matrices")
    missing = [key for key in target_keys if key not in matrices]
    if missing:
        raise KeyError(f"targets {missing} are not among the supplied matrices")
    scale = shared_channel_scale(matrices, channel_normalization) if scale is None else np.asarray(scale, float)
    balanced = {key: np.maximum(value, 0.0) / scale[:, None] for key, value in matrices.items()}

    source_W, _, source_metrics = fit_nmf_best(balanced[source_key], rank, n_init, seed)
    source_geometry = basis_geometry(source_W)

    targets: dict[str, Any] = {}
    for key in target_keys:
        target_matrix = balanced[key]
        own_rank = target_rank or rank
        own_W, _, own_metrics = fit_nmf_best(target_matrix, own_rank, n_init, seed + 31)
        heldout = heldout_reconstruction(source_W, target_matrix)
        novelty = synergy_novelty(source_W, own_W)
        match = match_synergies(source_W, own_W)
        targets[key] = {
            "trial_count": int(len(boundaries[key]) - 1),
            "own_rank": own_rank,
            # The gap between what the source basis explains and what the target's
            # own basis explains is the part that is specific to this action.
            "own_fit_global_vaf": own_metrics["global_vaf"],
            **heldout,
            "vaf_gap_to_own_fit": own_metrics["global_vaf"] - heldout["heldout_global_vaf"],
            "novelty": novelty,
            "matched_to_source": {
                "cosine_similarities": match["cosine_similarities"],
                "mean_cosine_similarity": match["mean_cosine_similarity"],
                "minimum_cosine_similarity": match["minimum_cosine_similarity"],
            },
            "own_basis_geometry": basis_geometry(own_W),
        }
    return {
        "schema_version": "emg_synergy_reuse_v1",
        "channel_normalization": channel_normalization,
        "shared_channel_scale": scale.tolist(),
        "source": {
            "key": source_key,
            "trial_count": int(len(boundaries[source_key]) - 1),
            "rank": rank,
            "global_vaf": source_metrics["global_vaf"],
            "local_vaf_fraction_ge_0_75": source_metrics["local_vaf_fraction_ge_0_75"],
            "basis_geometry": source_geometry,
        },
        "targets": targets,
    }


def shared_basis_recruitment(
    shared_W: np.ndarray,
    matrices: dict[str, np.ndarray],
    *,
    scale: np.ndarray,
    time_normalize_samples: int | None = None,
) -> dict[str, Any]:
    """How each action recruits one common set of synergies.

    Every action is projected onto the same fixed basis, so the coefficients are
    directly comparable: differences are in how much and when a synergy is used,
    not in what the synergy is.  Fitting each action its own basis cannot answer
    that, because the components would not refer to the same thing.
    """
    basis = np.asarray(shared_W, dtype=np.float64)
    rank = basis.shape[1]
    actions: dict[str, Any] = {}
    for key, value in matrices.items():
        balanced = np.maximum(np.asarray(value, dtype=np.float64), 0.0) / np.asarray(scale, dtype=np.float64)[:, None]
        coefficients, reconstruction = project_onto_basis(basis, balanced)
        global_vaf, _, _ = vaf_metrics(balanced, reconstruction)
        total = float(np.sum(coefficients)) or 1.0
        entry: dict[str, Any] = {
            "heldout_global_vaf": global_vaf,
            "mean_coefficient": coefficients.mean(axis=1).tolist(),
            # Share of total recruitment, so actions of different overall
            # intensity can be compared on which synergies they lean on.
            "recruitment_share": (coefficients.sum(axis=1) / total).tolist(),
            "peak_coefficient": coefficients.max(axis=1).tolist(),
        }
        if time_normalize_samples and coefficients.shape[1] % time_normalize_samples == 0:
            trials = coefficients.shape[1] // time_normalize_samples
            profile = coefficients.reshape(rank, trials, time_normalize_samples).mean(axis=1)
            entry["trials"] = trials
            entry["mean_activation_profile"] = profile.tolist()
            entry["peak_phase_percent"] = (
                profile.argmax(axis=1) * 100.0 / max(1, time_normalize_samples - 1)
            ).tolist()
        actions[key] = entry
    return {
        "schema_version": "emg_shared_basis_recruitment_v1",
        "rank": rank,
        "time_normalize_samples": time_normalize_samples,
        "actions": actions,
    }


def pairwise_reuse_matrix(
    matrices: dict[str, np.ndarray],
    *,
    rank: int,
    channel_normalization: str = "unit_variance",
    n_init: int = 30,
    seed: int = 20260720,
    scale: np.ndarray | None = None,
) -> dict[str, Any]:
    """Held-out VAF of every action's basis applied to every other action.

    The diagonal is a same-data fit and is an upper bound, not a result; the
    off-diagonal entries are what carry the reuse claim.
    """
    keys = list(matrices)
    scale = shared_channel_scale(matrices, channel_normalization) if scale is None else np.asarray(scale, float)
    balanced = {key: np.maximum(value, 0.0) / scale[:, None] for key, value in matrices.items()}
    bases = {key: fit_nmf_best(balanced[key], rank, n_init, seed + index * 17)[0] for index, key in enumerate(keys)}
    matrix = np.zeros((len(keys), len(keys)), dtype=np.float64)
    cosine = np.zeros_like(matrix)
    for row, source in enumerate(keys):
        for column, target in enumerate(keys):
            matrix[row, column] = heldout_reconstruction(bases[source], balanced[target])["heldout_global_vaf"]
            cosine[row, column] = match_synergies(bases[source], bases[target])["mean_cosine_similarity"]
    return {
        "schema_version": "emg_synergy_reuse_matrix_v1",
        "rank": rank,
        "channel_normalization": channel_normalization,
        "channel_scale": scale.tolist(),
        "actions": keys,
        "row_is_source_basis_column_is_target_data": True,
        "heldout_global_vaf": matrix.tolist(),
        "matched_basis_cosine": cosine.tolist(),
    }
