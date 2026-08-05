"""Run the whole-session muscle-synergy study for one participant session.

The per-action datasets and the pooled dataset are both built through
``build_synergy_dataset`` and fitted through ``extract_synergy`` so the study
inherits their profile, preprocessing and channel-order gates instead of
re-deriving them.  On top of that this driver adds the two things a single
extraction cannot answer: how many synergies survive repeated resampling of the
trials, and which of an action's synergies are shared with the rest of the
session rather than specific to that action.

Usage::

    python scripts/run_synergy_study.py \
        --dataset-root data --participant P002 \
        --output-dir data/analysis/P002_S20260721_A_study
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from emg.dataset import build_synergy_dataset
from emg.storage import atomic_write_json, read_json
from emg.synergy import channel_scale, extract_synergy, fit_nmf_best, match_synergies, trial_split_half
from emg.synergy_reuse import (
    basis_geometry,
    bootstrap_stability,
    initialization_stability,
    pairwise_reuse_matrix,
    reuse_analysis,
    shared_basis_recruitment,
    shared_channel_scale,
    within_action_heldout,
)

# Below this an action cannot support a split-half check, so its basis is
# reported as a descriptive fit only and never used to claim a synergy count.
MINIMUM_TRIALS_FOR_ACTION_FIT = 3


def _muscle_labels(metadata: dict[str, Any]) -> list[str]:
    channels = metadata["channel_profile_snapshot"]["channels"]
    return [f"S{item['sensor_id']:02d} {item['side'][0].upper()}-{item['abbreviation']}" for item in channels]


def _available_actions(dataset_metadata: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trial in dataset_metadata["trials"]:
        counts[trial["action_id"]] = counts.get(trial["action_id"], 0) + 1
    return counts


def _fit(
    dataset_root: Path,
    output_dir: Path,
    tag: str,
    *,
    profile: str,
    protocol: str,
    participant: str,
    crop_mode: str,
    time_normalize_samples: int | None,
    action_ids: list[str] | None,
    channel_normalization: str,
    k_max: int,
    n_init: int,
    seed: int,
    split_half_repeats: int,
) -> dict[str, Any] | None:
    dataset_path = output_dir / "datasets" / f"{tag}.npz"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    build_synergy_dataset(
        dataset_root,
        dataset_path,
        profile_id=profile,
        protocol_id=protocol,
        only_valid=True,
        scope="all",
        action_ids=action_ids,
        participant_ids=[participant],
        crop_mode=crop_mode,
        time_normalize_samples=time_normalize_samples,
    )
    dataset_metadata = read_json(dataset_path.with_suffix(".json"))
    trial_count = len(dataset_metadata["trials"])
    if trial_count < MINIMUM_TRIALS_FOR_ACTION_FIT:
        print(f"  skip {tag}: only {trial_count} analysis-ready trial(s)")
        return None
    fit_dir = output_dir / "fits" / tag
    print(f"  fitting {tag} ({trial_count} trials)")
    extract_synergy(
        dataset_path,
        fit_dir,
        k_min=1,
        k_max=k_max,
        n_init=n_init,
        seed=seed,
        include_h=True,
        calculate_stability=True,
        channel_normalization=channel_normalization,
        split_half_repeats=split_half_repeats,
    )
    with np.load(fit_dir / "synergy_basis.npz", allow_pickle=False) as payload:
        basis = np.asarray(payload["W"], dtype=np.float64)
        activation = np.asarray(payload["H"], dtype=np.float64)
    return {
        "tag": tag,
        "trial_count": trial_count,
        "dataset_path": str(dataset_path),
        "fit_dir": str(fit_dir),
        "W": basis,
        "H": activation,
        "metrics": read_json(fit_dir / "metrics.json"),
        "metadata": read_json(fit_dir / "synergy_metadata.json"),
        "dataset_metadata": dataset_metadata,
    }


def stability_sweep(
    dataset_path: Path,
    *,
    channel_normalization: str,
    k_max: int,
    n_init: int,
    seed: int,
    repeats: int,
    reproducible_threshold: float = 0.80,
) -> dict[str, Any]:
    """How many synergies survive resampling the trials, for each K.

    Reconstruction alone cannot answer this: VAF rises monotonically with K, so
    a threshold on it will always reward the largest model tried.  A synergy
    that does not reappear when the trials are split a different way describes
    this particular recording rather than the participant, so the count that can
    be defended is the largest K whose synergies keep matching across splits.
    """
    with np.load(dataset_path, allow_pickle=False) as payload:
        matrix = np.asarray(payload["V"], dtype=np.float64)
        boundaries = np.asarray(payload["trial_boundaries"])
    scale = channel_scale(matrix, channel_normalization)
    balanced = matrix / scale[:, None]
    rows: list[dict[str, Any]] = []
    for k in range(2, min(k_max, matrix.shape[0]) + 1):
        result = trial_split_half(balanced, k, n_init, seed + k * 977, boundaries, repeats)
        if not result.get("available"):
            rows.append({"k": k, "available": False, "reason": result.get("reason")})
            continue
        per_run_min = [item["minimum_cosine_similarity"] for item in result["runs"]]
        rows.append(
            {
                "k": k,
                "available": True,
                "repeats": result["repeats"],
                "mean_cosine_similarity": result["mean_cosine_similarity"],
                "median_minimum_cosine_similarity": float(np.median(per_run_min)),
                "mean_minimum_cosine_similarity": result["mean_minimum_cosine_similarity"],
                "worst_minimum_cosine_similarity": result["worst_minimum_cosine_similarity"],
                "fraction_of_repeats_all_synergies_ge_0_80": result["fraction_of_repeats_all_synergies_ge_0_80"],
            }
        )
    supported = [
        row["k"]
        for row in rows
        if row.get("available") and row["median_minimum_cosine_similarity"] >= reproducible_threshold
    ]
    return {
        "reproducible_threshold": reproducible_threshold,
        "criterion": "largest K whose least-reproducible synergy still matches across the median split",
        # Recorded because agreement is bounded below by the optimiser: too few
        # restarts leaves the halves in worse local minima and understates
        # reproducibility, so these numbers are only comparable at equal budget.
        "split_half_n_init": n_init,
        "split_half_repeats": repeats,
        "largest_reproducible_k": max(supported) if supported else None,
        "by_k": rows,
    }


def stability_suite(
    dataset_path: Path,
    *,
    rank: int,
    channel_normalization: str,
    n_init: int,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    """The remaining stability views a single fit does not provide.

    Split-half and cross-trial already live in the fit artifact.  Initialization
    agreement separates optimiser noise from data noise, bootstrap resampling
    says how much the basis depends on which repetitions were recorded, and the
    basis geometry says whether the columns actually span the rank they claim.
    """
    with np.load(dataset_path, allow_pickle=False) as payload:
        matrix = np.asarray(payload["V"], dtype=np.float64)
        boundaries = np.asarray(payload["trial_boundaries"])
    balanced = np.maximum(matrix, 0.0) / channel_scale(matrix, channel_normalization)[:, None]
    reference, _, _ = fit_nmf_best(balanced, rank, n_init, seed)
    return {
        "rank": rank,
        "initialization": initialization_stability(balanced, rank, n_init, seed, restarts=6),
        "bootstrap": bootstrap_stability(balanced, reference, rank, n_init, seed, boundaries, repeats=repeats),
        "within_action_heldout": within_action_heldout(balanced, boundaries, rank, n_init, seed, repeats=repeats),
        "basis_geometry": basis_geometry(reference),
    }


def _load_matrix(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(dataset_path, allow_pickle=False) as payload:
        return np.asarray(payload["V"], dtype=np.float64), np.asarray(payload["trial_boundaries"])


def build_reuse_section(
    dataset_root: Path,
    output_dir: Path,
    per_action_keys: list[str],
    *,
    profile: str,
    protocol: str,
    participant: str,
    crop_mode: str,
    time_normalize_samples: int | None,
    channel_normalization: str,
    reuse_rank: int,
    n_init: int,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    """Do the complete strokes reuse the basic movements' synergies?

    The basic-action pool is fitted once and then held fixed while each complete
    stroke's coefficients are solved, so the reported VAF is what the basic
    structure already explains.  It is reported beside each action's own
    within-action held-out VAF, because no basis reaches a refit's VAF even on
    the same action and the cross-action number is only meaningful against that.
    """
    scopes: dict[str, Path] = {}
    for scope in ("primitive", "complete"):
        path = output_dir / "datasets" / f"scope_{scope}.npz"
        build_synergy_dataset(
            dataset_root,
            path,
            profile_id=profile,
            protocol_id=protocol,
            only_valid=True,
            scope=scope,
            action_ids=None,
            participant_ids=[participant],
            crop_mode=crop_mode,
            time_normalize_samples=time_normalize_samples,
        )
        scopes[scope] = path

    complete_metadata = read_json(scopes["complete"].with_suffix(".json"))
    complete_actions = sorted({trial["action_id"] for trial in complete_metadata["trials"]})
    targets = [action for action in complete_actions if action in per_action_keys]

    matrices: dict[str, np.ndarray] = {}
    boundaries: dict[str, np.ndarray] = {}
    matrices["primitive_pool"], boundaries["primitive_pool"] = _load_matrix(scopes["primitive"])
    for action in targets:
        matrices[action], boundaries[action] = _load_matrix(output_dir / "datasets" / f"action_{action}.npz")

    section: dict[str, Any] = {
        "basic_action_pool": {
            "trial_count": int(len(boundaries["primitive_pool"]) - 1),
            "actions": sorted({t["action_id"] for t in read_json(scopes["primitive"].with_suffix(".json"))["trials"]}),
        },
        "complete_actions_available": complete_actions,
        "complete_actions_analysed": targets,
    }
    if not targets:
        section["available"] = False
        section["reason"] = "no complete action had enough analysis-ready trials to fit"
        return section

    section["available"] = True
    # One scale for the whole session, so a comparison's numbers do not move
    # when a different subset of actions is compared.
    pooled_matrix, _ = _load_matrix(output_dir / "datasets" / "pooled_all_actions.npz")
    scale = shared_channel_scale({"session": pooled_matrix}, channel_normalization)
    section["channel_scale_reference"] = "pooled_all_actions"
    section["basic_to_complete"] = reuse_analysis(
        matrices,
        boundaries,
        source_key="primitive_pool",
        target_keys=targets,
        rank=reuse_rank,
        channel_normalization=channel_normalization,
        n_init=n_init,
        seed=seed,
        scale=scale,
    )
    ceilings: dict[str, Any] = {}
    for key in ["primitive_pool", *targets]:
        balanced = np.maximum(matrices[key], 0.0) / scale[:, None]
        ceilings[key] = within_action_heldout(balanced, boundaries[key], reuse_rank, n_init, seed, repeats=repeats)
    section["within_action_heldout_ceiling"] = ceilings

    per_action = {key: _load_matrix(output_dir / "datasets" / f"action_{key}.npz")[0] for key in per_action_keys}
    # One basis fitted on the whole session, then every action projected onto it,
    # so "which synergies does this action use, and when" is a comparison of the
    # same components rather than of separately named ones.
    shared_basis, _, _ = fit_nmf_best(
        np.maximum(pooled_matrix, 0.0) / scale[:, None], reuse_rank, n_init, seed
    )
    section["shared_basis_recruitment"] = shared_basis_recruitment(
        shared_basis,
        per_action,
        scale=scale,
        time_normalize_samples=time_normalize_samples,
    )
    section["pairwise_reuse_matrix"] = pairwise_reuse_matrix(
        per_action,
        rank=reuse_rank,
        channel_normalization=channel_normalization,
        n_init=n_init,
        seed=seed,
        scale=scale,
    )
    return section


def _stability_selected_k(metrics: dict[str, Any]) -> dict[str, Any]:
    """Report the fitted K alongside what the split-half check actually supports.

    ``extract_synergy`` picks K from reconstruction thresholds alone.  A basis
    that reconstructs well but does not reappear when the trials are resampled
    is a description of this particular recording, not of the participant, so
    the two numbers are reported separately rather than reconciled here.
    """
    split_half = metrics.get("stability", {}).get("split_half", {})
    return {
        "selected_k": metrics["selected_k"],
        "selection_is_fallback_largest_k": metrics.get("selection_is_fallback_largest_k"),
        "split_half_available": bool(split_half.get("available")),
        "mean_cosine_similarity": split_half.get("mean_cosine_similarity"),
        "worst_minimum_cosine_similarity": split_half.get("worst_minimum_cosine_similarity"),
        "fraction_of_repeats_all_synergies_ge_0_80": split_half.get("fraction_of_repeats_all_synergies_ge_0_80"),
    }


def run_study(
    dataset_root: Path,
    output_dir: Path,
    *,
    participant: str,
    profile: str,
    protocol: str,
    crop_mode: str,
    time_normalize_samples: int | None,
    channel_normalization: str,
    k_max: int,
    n_init: int,
    seed: int,
    split_half_repeats: int,
    reuse_rank: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "profile": profile,
        "protocol": protocol,
        "participant": participant,
        "crop_mode": crop_mode,
        "time_normalize_samples": time_normalize_samples,
        "channel_normalization": channel_normalization,
        "k_max": k_max,
        "n_init": n_init,
        "seed": seed,
        "split_half_repeats": split_half_repeats,
    }
    print("Pooled fit across every analysis-ready trial in the session")
    pooled = _fit(dataset_root, output_dir, "pooled_all_actions", action_ids=None, **common)
    if pooled is None:
        raise RuntimeError("The pooled dataset has too few analysis-ready trials to fit")
    labels = _muscle_labels(pooled["dataset_metadata"])
    action_counts = _available_actions(pooled["dataset_metadata"])
    print(f"Per-action fits (actions with >= {MINIMUM_TRIALS_FOR_ACTION_FIT} trials)")
    per_action: dict[str, dict[str, Any]] = {}
    for action_id, count in sorted(action_counts.items(), key=lambda item: -item[1]):
        if count < MINIMUM_TRIALS_FOR_ACTION_FIT:
            print(f"  skip {action_id}: {count} analysis-ready trial(s)")
            continue
        result = _fit(dataset_root, output_dir, f"action_{action_id}", action_ids=[action_id], **common)
        if result is not None:
            per_action[action_id] = result

    print("Stability sweep over K")
    sweeps = {"pooled_all_actions": stability_sweep(
        Path(pooled["dataset_path"]),
        channel_normalization=channel_normalization,
        k_max=k_max,
        n_init=n_init,
        seed=seed,
        repeats=split_half_repeats,
    )}
    for action_id, result in per_action.items():
        sweeps[action_id] = stability_sweep(
            Path(result["dataset_path"]),
            channel_normalization=channel_normalization,
            k_max=k_max,
            n_init=n_init,
            seed=seed,
            repeats=split_half_repeats,
        )
        print(f"  {action_id}: largest reproducible K = {sweeps[action_id]['largest_reproducible_k']}")

    print("Stability suite (initialization / bootstrap / geometry)")
    suites: dict[str, Any] = {}
    for action_id, result in per_action.items():
        supported = sweeps[action_id]["largest_reproducible_k"]
        if supported is None:
            suites[action_id] = {"available": False, "reason": "no reproducible K to characterise"}
            continue
        suites[action_id] = stability_suite(
            Path(result["dataset_path"]),
            rank=supported,
            channel_normalization=channel_normalization,
            n_init=n_init,
            seed=seed,
            repeats=split_half_repeats,
        )
        print(f"  {action_id}: at K={supported}")

    print("Basic-to-complete synergy reuse")
    reuse = build_reuse_section(
        dataset_root,
        output_dir,
        list(per_action),
        profile=profile,
        protocol=protocol,
        participant=participant,
        crop_mode=crop_mode,
        time_normalize_samples=time_normalize_samples,
        channel_normalization=channel_normalization,
        reuse_rank=reuse_rank,
        n_init=n_init,
        seed=seed,
        repeats=split_half_repeats,
    )

    shared = {}
    for action_id, result in per_action.items():
        match = match_synergies(pooled["W"], result["W"])
        shared[action_id] = {
            "matched_against": "pooled_all_actions",
            "cosine_similarities": match["cosine_similarities"],
            "mean_cosine_similarity": match["mean_cosine_similarity"],
            "minimum_cosine_similarity": match["minimum_cosine_similarity"],
            "pooled_synergy_indices": match["reference_indices"],
            "action_synergy_indices": match["candidate_indices"],
            "n_shared_ge_0_80": int(sum(1 for value in match["cosine_similarities"] if value >= 0.80)),
        }

    summary = {
        "schema_version": "emg_synergy_study_v1",
        "participant_id": participant,
        "profile_id": profile,
        "protocol_id": protocol,
        "crop_mode": crop_mode,
        "crop_is_exploratory": bool(pooled["dataset_metadata"]["exploratory"]),
        "time_normalize_samples": time_normalize_samples,
        "channel_normalization": channel_normalization,
        "reuse_rank": reuse_rank,
        "muscle_labels": labels,
        "analysis_ready_trials_per_action": action_counts,
        "pooled": {
            "trial_count": pooled["trial_count"],
            "matrix_shape": pooled["dataset_metadata"]["matrix_shape"],
            **_stability_selected_k(pooled["metrics"]),
            "global_vaf": pooled["metadata"]["global_vaf"],
            "recorded_unit_global_vaf": pooled["metadata"]["recorded_unit_global_vaf"],
            "fit_dir": pooled["fit_dir"],
        },
        "per_action": {
            action_id: {
                "trial_count": result["trial_count"],
                **_stability_selected_k(result["metrics"]),
                "global_vaf": result["metadata"]["global_vaf"],
                "recorded_unit_global_vaf": result["metadata"]["recorded_unit_global_vaf"],
                "fit_dir": result["fit_dir"],
            }
            for action_id, result in per_action.items()
        },
        "action_to_pooled_similarity": shared,
        "stability_sweep": sweeps,
        "stability_suite": suites,
        "synergy_reuse": reuse,
        "vaf_by_k": {
            "pooled_all_actions": [
                {
                    "k": item["k"],
                    "global_vaf": item["global_vaf"],
                    "recorded_unit_global_vaf": item["recorded_unit_global_vaf"],
                    "worst_local_vaf": float(np.min(item["local_vaf"])),
                    "fraction_local_vaf_ge_0_75": item["local_vaf_fraction_ge_0_75"],
                }
                for item in pooled["metrics"]["k_search"]
            ],
            **{
                action_id: [
                    {
                        "k": item["k"],
                        "global_vaf": item["global_vaf"],
                        "recorded_unit_global_vaf": item["recorded_unit_global_vaf"],
                        "worst_local_vaf": float(np.min(item["local_vaf"])),
                        "fraction_local_vaf_ge_0_75": item["local_vaf_fraction_ge_0_75"],
                    }
                    for item in result["metrics"]["k_search"]
                ]
                for action_id, result in per_action.items()
            },
        },
    }
    summary_path = output_dir / "synergy_study_summary.json"
    atomic_write_json(summary_path, summary)
    print(f"Study summary saved: {summary_path}")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--profile", default="badminton_synergy_16_v2")
    parser.add_argument("--protocol", default="badminton_primitive_protocol_v1")
    parser.add_argument(
        "--crop-mode",
        choices=("annotated_movement_events", "software_cue_exploratory", "full_trial"),
        default="software_cue_exploratory",
    )
    parser.add_argument("--time-normalize-samples", type=int, default=101)
    parser.add_argument(
        "--channel-normalization",
        choices=("none", "unit_max", "unit_variance"),
        default="unit_variance",
    )
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--split-half-repeats", type=int, default=30)
    parser.add_argument(
        "--reuse-rank",
        type=int,
        default=3,
        help="Rank used for the basic-to-complete reuse comparison; pick one the stability sweep supports.",
    )
    parser.add_argument(
        "--stability-sweep-only",
        action="store_true",
        help="Recompute the K stability sweep for an existing study directory without refitting.",
    )
    return parser


def refresh_stability_sweep(output_dir: Path, *, k_max: int, n_init: int, seed: int, repeats: int) -> Path:
    """Recompute the sweep against an existing study without refitting the bases.

    The sweep only reads the built datasets, so it can be re-run at a different
    repeat count without disturbing the fitted artifacts it is reported beside.
    """
    summary_path = output_dir / "synergy_study_summary.json"
    summary = read_json(summary_path)
    normalization = summary["channel_normalization"]
    sweeps: dict[str, Any] = {}
    for tag in ["pooled_all_actions", *summary["per_action"]]:
        dataset_path = output_dir / "datasets" / (f"{tag}.npz" if tag == "pooled_all_actions" else f"action_{tag}.npz")
        sweeps[tag] = stability_sweep(
            dataset_path,
            channel_normalization=normalization,
            k_max=k_max,
            n_init=n_init,
            seed=seed,
            repeats=repeats,
        )
        print(f"  {tag}: largest reproducible K = {sweeps[tag]['largest_reproducible_k']}")
    summary["stability_sweep"] = sweeps
    atomic_write_json(summary_path, summary)
    print(f"Study summary updated: {summary_path}")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stability_sweep_only:
        refresh_stability_sweep(
            args.output_dir,
            k_max=args.k_max,
            n_init=args.n_init,
            seed=args.seed,
            repeats=args.split_half_repeats,
        )
        return 0
    run_study(
        args.dataset_root,
        args.output_dir,
        participant=args.participant,
        profile=args.profile,
        protocol=args.protocol,
        crop_mode=args.crop_mode,
        time_normalize_samples=args.time_normalize_samples,
        channel_normalization=args.channel_normalization,
        k_max=args.k_max,
        n_init=args.n_init,
        seed=args.seed,
        split_half_repeats=args.split_half_repeats,
        reuse_rank=args.reuse_rank,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
