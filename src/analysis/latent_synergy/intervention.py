"""Construct latent perturbations and summarize decoder-space effect sizes.

These offline summaries are descriptive sensitivity evidence, not causal
environment-rollout evidence.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.latent_muscle.phase_contract import (
    FOREHAND_PHASE_NAMES,
    phase_items,
)

PHASE_NAMES = FOREHAND_PHASE_NAMES


def build_intervention_latents(
    base_latents: np.ndarray,
    directions: np.ndarray,
    *,
    epsilons: Sequence[float] = (-1.0, -0.5, 0.5, 1.0),
    normalize_directions: bool = True,
) -> np.ndarray:
    """Return ``[sample, direction, epsilon, latent_dim]`` perturbations."""

    z = _finite_matrix("base_latents", base_latents)
    v = _finite_matrix("directions", directions)
    if v.shape[1] != z.shape[1]:
        raise ValueError("intervention directions must match latent_dim")
    epsilon = np.asarray(tuple(float(value) for value in epsilons), dtype=np.float64)
    if epsilon.size == 0 or not np.all(np.isfinite(epsilon)) or np.any(epsilon == 0.0):
        raise ValueError("intervention epsilons must be finite, non-zero, and non-empty")
    if normalize_directions:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("intervention directions cannot contain a zero vector")
        v = v / norms
    return z[:, None, None, :] + epsilon[None, None, :, None] * v[None, :, None, :]


def evaluate_interventions(
    decoder: Callable[[np.ndarray, np.ndarray], Any],
    states: np.ndarray,
    base_latents: np.ndarray,
    directions: np.ndarray,
    *,
    epsilons: Sequence[float] = (-1.0, -0.5, 0.5, 1.0),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Evaluate a NumPy-compatible decoder callable without assuming its backend.

    The callable may return an array (recorded as ``action``), a mapping, or a
    NamedTuple such as ``SynergyDecoderOutput``.  This helper only evaluates
    the supplied decoder callable. Environment outcomes must instead pass the
    separately sealed paired-rollout artifact contract.
    """

    state = _finite_matrix("states", states)
    z = _finite_matrix("base_latents", base_latents)
    if state.shape[0] != z.shape[0]:
        raise ValueError("states and base_latents must have equal sample counts")
    perturbed_z = build_intervention_latents(z, directions, epsilons=epsilons)
    baseline = _output_mapping(decoder(state, z))
    flat_z = perturbed_z.reshape((-1, z.shape[1]))
    repeated_state = np.repeat(state, perturbed_z.shape[1] * perturbed_z.shape[2], axis=0)
    perturbed_flat = _output_mapping(decoder(repeated_state, flat_z))
    perturbed: dict[str, np.ndarray] = {}
    for key, value in perturbed_flat.items():
        if value.shape[0] != flat_z.shape[0]:
            raise ValueError(f"decoder intervention output {key!r} has wrong sample dimension")
        perturbed[key] = value.reshape((z.shape[0], perturbed_z.shape[1], perturbed_z.shape[2], *value.shape[1:]))
    return baseline, perturbed


def summarize_intervention_effects(
    baseline: Mapping[str, np.ndarray],
    perturbed: Mapping[str, np.ndarray],
    *,
    epsilons: Sequence[float],
    direction_names: Sequence[str] | None = None,
    phase_ids: np.ndarray | None = None,
    require_metrics: Sequence[str] = (),
    require_all_phases: bool = False,
    phase_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not baseline or not perturbed:
        raise ValueError("intervention baseline and perturbed outputs must be non-empty")
    missing = sorted({str(key) for key in require_metrics} - set(baseline))
    missing += sorted({str(key) for key in require_metrics} - set(perturbed))
    if missing:
        raise ValueError(f"intervention evidence is missing required metrics: {sorted(set(missing))}")
    shared = sorted(set(baseline) & set(perturbed))
    if not shared:
        raise ValueError("baseline and perturbed intervention outputs share no metrics")
    epsilon = np.asarray(tuple(float(value) for value in epsilons), dtype=np.float64)
    first = np.asarray(perturbed[shared[0]])
    if first.ndim < 3 or first.shape[0] <= 0 or first.shape[1] <= 0:
        raise ValueError("perturbed outputs must be [sample, direction, epsilon, ...]")
    num_samples, num_directions, num_epsilons = first.shape[:3]
    if epsilon.shape != (num_epsilons,) or not np.all(np.isfinite(epsilon)):
        raise ValueError("epsilons must match the intervention output epsilon axis")
    names = (
        tuple(f"direction_{index}" for index in range(num_directions))
        if direction_names is None
        else tuple(str(name) for name in direction_names)
    )
    if len(names) != num_directions or len(set(names)) != len(names):
        raise ValueError("direction_names must uniquely match the direction axis")
    metric_deltas: dict[str, np.ndarray] = {}
    for key in shared:
        base = np.asarray(baseline[key], dtype=np.float64)
        changed = np.asarray(perturbed[key], dtype=np.float64)
        if base.shape[0] != num_samples or changed.shape[:3] != first.shape[:3]:
            raise ValueError(f"intervention metric {key!r} has inconsistent leading dimensions")
        if changed.shape[3:] != base.shape[1:]:
            raise ValueError(f"intervention metric {key!r} feature dimensions differ")
        if not np.all(np.isfinite(base)) or not np.all(np.isfinite(changed)):
            raise ValueError(f"intervention metric {key!r} contains non-finite values")
        metric_deltas[key] = changed - base[:, None, None, ...]
    report: dict[str, Any] = {
        "schema_version": "latent_synergy_intervention_v1",
        "num_samples": int(num_samples),
        "direction_names": list(names),
        "epsilons": epsilon.tolist(),
        "metrics": _summarize_delta_mapping(metric_deltas, names, epsilon),
    }
    if phase_ids is not None:
        phases_and_names = phase_items(phase_contract)
        phases = _validated_phases(
            phase_ids,
            num_samples,
            known_phase_ids={phase_id for phase_id, _name in phases_and_names},
        )
        by_phase: dict[str, Any] = {}
        absent: list[str] = []
        for phase_id, phase_name in phases_and_names:
            selected = phases == phase_id
            if not np.any(selected):
                absent.append(phase_name)
                continue
            by_phase[phase_name] = {
                "num_samples": int(np.sum(selected)),
                "metrics": _summarize_delta_mapping(
                    {key: value[selected] for key, value in metric_deltas.items()},
                    names,
                    epsilon,
                ),
            }
        if require_all_phases and absent:
            raise ValueError(f"intervention report is missing phases: {absent}")
        report["by_phase"] = by_phase
        report["missing_phases"] = absent
    elif require_all_phases:
        raise ValueError("require_all_phases=True requires phase_ids")
    return report


def _summarize_delta_mapping(
    deltas: Mapping[str, np.ndarray],
    names: Sequence[str],
    epsilons: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, delta in deltas.items():
        feature_axes = tuple(range(3, delta.ndim))
        squared = np.square(delta)
        rms_by_sample = np.sqrt(np.mean(squared, axis=feature_axes) if feature_axes else squared)
        signed_by_sample = np.mean(delta, axis=feature_axes) if feature_axes else delta
        by_direction: dict[str, Any] = {}
        for direction_index, name in enumerate(names):
            by_direction[name] = {
                str(float(epsilons[epsilon_index])): {
                    "delta_rms_mean": float(np.mean(rms_by_sample[:, direction_index, epsilon_index])),
                    "delta_rms_std": float(np.std(rms_by_sample[:, direction_index, epsilon_index])),
                    "delta_signed_mean": float(np.mean(signed_by_sample[:, direction_index, epsilon_index])),
                }
                for epsilon_index in range(len(epsilons))
            }
        result[key] = by_direction
    return result


def _output_mapping(value: Any) -> dict[str, np.ndarray]:
    if isinstance(value, Mapping):
        mapping = dict(value)
    elif hasattr(value, "_asdict"):
        mapping = dict(value._asdict())
    else:
        mapping = {"action": value}
    result = {str(key): np.asarray(item) for key, item in mapping.items()}
    if not result or any(array.ndim == 0 for array in result.values()):
        raise ValueError("decoder outputs must be non-empty arrays with a sample axis")
    return result


def _validated_phases(
    value: np.ndarray,
    size: int,
    *,
    known_phase_ids: set[int] | None = None,
) -> np.ndarray:
    phases = np.asarray(value)
    if phases.shape != (size,) or not np.all(np.isfinite(phases)) or not np.all(phases == np.floor(phases)):
        raise ValueError("phase_ids must be a finite integer vector matching samples")
    phases = phases.astype(np.int64)
    known = set(range(len(PHASE_NAMES))) if known_phase_ids is None else set(known_phase_ids)
    unknown = sorted(set(phases.tolist()) - known)
    if unknown:
        raise ValueError(f"phase_ids contain unknown values: {unknown}")
    return phases


def _finite_matrix(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) <= 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite non-empty rank-2 matrix")
    return array


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("build", "summarize"), required=True)
    parser.add_argument("--epsilon", type=float, nargs="+", default=[-1.0, -0.5, 0.5, 1.0])
    parser.add_argument("--latent-key", default="base_latents")
    parser.add_argument("--direction-key", default="directions")
    parser.add_argument("--phase-key", default=None)
    parser.add_argument("--require-metric", action="append", default=[])
    return parser


def main() -> int:
    args = _parser().parse_args()
    with np.load(args.input_npz, allow_pickle=False) as data:
        if args.mode == "build":
            if args.latent_key not in data.files or args.direction_key not in data.files:
                raise ValueError("build mode requires base latent and direction arrays")
            output = build_intervention_latents(data[args.latent_key], data[args.direction_key], epsilons=args.epsilon)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.output,
                intervention_latents=output,
                epsilons=np.asarray(args.epsilon, dtype=np.float64),
            )
            return 0
        baseline = {
            key.removeprefix("baseline_"): np.asarray(data[key]) for key in data.files if key.startswith("baseline_")
        }
        perturbed = {
            key.removeprefix("perturbed_"): np.asarray(data[key]) for key in data.files if key.startswith("perturbed_")
        }
        report = summarize_intervention_effects(
            baseline,
            perturbed,
            epsilons=args.epsilon,
            phase_ids=None if args.phase_key is None else data[args.phase_key],
            require_metrics=args.require_metric,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
