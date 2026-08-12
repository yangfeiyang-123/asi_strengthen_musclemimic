"""Rotation-invariant stability analysis across latent random seeds."""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from analysis.latent_synergy.jacobian_alignment import subspace_alignment
from analysis.latent_synergy.representation_similarity import (
    linear_cka,
    procrustes_aligned_similarity,
)


def cross_seed_report(
    representations: Mapping[str, np.ndarray],
    *,
    jacobian_spans: Mapping[str, np.ndarray] | None = None,
    causal_effects: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    normalized_inputs = {str(name): value for name, value in representations.items()}
    if len(normalized_inputs) != len(representations):
        raise ValueError("cross-seed names collide after string normalization")
    names = sorted(normalized_inputs)
    if len(names) < 2:
        raise ValueError("cross-seed analysis requires at least two seeds")
    matrices = {name: _finite_matrix(f"representation[{name}]", normalized_inputs[name]) for name in names}
    sample_counts = {matrix.shape[0] for matrix in matrices.values()}
    if len(sample_counts) != 1:
        raise ValueError("cross-seed representations must use the same ordered samples")
    pairs: list[dict[str, Any]] = []
    for left, right in itertools.combinations(names, 2):
        item: dict[str, Any] = {
            "seed_a": left,
            "seed_b": right,
            "linear_cka": linear_cka(matrices[left], matrices[right]),
            "procrustes_aligned_similarity": procrustes_aligned_similarity(matrices[left], matrices[right]),
        }
        if jacobian_spans is not None:
            if left not in jacobian_spans or right not in jacobian_spans:
                raise ValueError("jacobian_spans must contain every representation seed")
            a = _finite_matrix(f"jacobian_span[{left}]", jacobian_spans[left])
            b = _finite_matrix(f"jacobian_span[{right}]", jacobian_spans[right])
            item["jacobian_span"] = subspace_alignment(a, b)
        if causal_effects is not None:
            if left not in causal_effects or right not in causal_effects:
                raise ValueError("causal_effects must contain every representation seed")
            effect_a = np.asarray(causal_effects[left], dtype=np.float64).reshape(-1)
            effect_b = np.asarray(causal_effects[right], dtype=np.float64).reshape(-1)
            if (
                effect_a.shape != effect_b.shape
                or effect_a.size == 0
                or not np.all(np.isfinite(effect_a))
                or not np.all(np.isfinite(effect_b))
            ):
                raise ValueError("paired causal effect vectors must be finite and shape-aligned")
            denominator = np.linalg.norm(effect_a) * np.linalg.norm(effect_b)
            if denominator <= 1e-12:
                raise ValueError("cross-seed causal effect cosine is undefined for zero effects")
            item["causal_effect_cosine_similarity"] = float(np.dot(effect_a, effect_b) / denominator)
        pairs.append(item)
    report: dict[str, Any] = {
        "schema_version": "latent_synergy_cross_seed_v1",
        "seeds": names,
        "num_pairs": len(pairs),
        "num_aligned_samples": next(iter(sample_counts)),
        "pairs": pairs,
        "linear_cka_mean": float(np.mean([item["linear_cka"] for item in pairs])),
        "linear_cka_std": float(np.std([item["linear_cka"] for item in pairs])),
        "procrustes_similarity_mean": float(np.mean([item["procrustes_aligned_similarity"] for item in pairs])),
    }
    if jacobian_spans is not None:
        report["jacobian_projection_score_mean"] = float(
            np.mean([item["jacobian_span"]["projection_score"] for item in pairs])
        )
    if causal_effects is not None:
        report["causal_effect_cosine_mean"] = float(
            np.mean([item["causal_effect_cosine_similarity"] for item in pairs])
        )
    return report


def _finite_matrix(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) <= 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite non-empty rank-2 matrix")
    return array


def _seed_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("seed input must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("seed input must be NAME=PATH")
    return name.strip(), Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-npz", type=_seed_path, action="append", required=True)
    parser.add_argument("--representation-key", default="latents")
    parser.add_argument("--jacobian-span-key", default=None)
    parser.add_argument("--causal-effect-key", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    representations: dict[str, np.ndarray] = {}
    jacobians: dict[str, np.ndarray] = {}
    effects: dict[str, np.ndarray] = {}
    for name, path in args.seed_npz:
        if name in representations:
            raise ValueError(f"duplicate seed name {name!r}")
        with np.load(path, allow_pickle=False) as data:
            if args.representation_key not in data.files:
                raise ValueError(f"{path} is missing {args.representation_key!r}")
            representations[name] = np.asarray(data[args.representation_key])
            if args.jacobian_span_key is not None:
                jacobians[name] = np.asarray(data[args.jacobian_span_key])
            if args.causal_effect_key is not None:
                effects[name] = np.asarray(data[args.causal_effect_key])
    report = cross_seed_report(
        representations,
        jacobian_spans=None if args.jacobian_span_key is None else jacobians,
        causal_effects=None if args.causal_effect_key is None else effects,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
