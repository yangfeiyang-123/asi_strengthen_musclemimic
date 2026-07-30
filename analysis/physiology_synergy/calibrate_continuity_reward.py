"""Calibrate a fixed continuity-reward coefficient from held-out baseline steps.

This module consumes evidence; it never runs a policy and never invents a
coefficient.  The output pins the baseline checkpoint, promotion, rollout,
environment, taxonomy, graph and held-out split used for calibration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

BASELINE_ROLLOUT_SCHEMA_VERSION = "fascicle_continuity_baseline_rollout_v2"
CALIBRATION_SCHEMA_VERSION = "fascicle_continuity_reward_calibration_v2"
PENALTY_BUDGET_FRACTIONS = (0.005, 0.01, 0.02)
MINIMUM_CALIBRATION_STEPS = 32
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def baseline_rollout_fingerprint(payload: Mapping[str, Any]) -> str:
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("artifact_fingerprint", None)
    return _json_sha256(unsigned)


def calibration_fingerprint(payload: Mapping[str, Any]) -> str:
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("calibration_fingerprint", None)
    return _json_sha256(unsigned)


def validate_baseline_rollout_identity(value: Any) -> dict[str, Any]:
    """Validate the immutable policy/environment/split identity for a rollout."""

    identity = _mapping(value, "baseline identity")
    _exact_keys(
        identity,
        {
            "run_id",
            "action_mode",
            "config_hash",
            "checkpoint_fingerprint",
            "promotion_fingerprint",
            "rollout_manifest_fingerprint",
            "environment_fingerprint",
            "heldout_split_fingerprint",
            "taxonomy_fingerprint",
            "continuity_graph_fingerprint",
        },
        "baseline identity",
    )
    return {
        "run_id": _text(identity["run_id"], "run_id"),
        "action_mode": _action_mode(identity["action_mode"]),
        "config_hash": _text(identity["config_hash"], "config_hash"),
        **{
            field: _sha256(identity[field], field)
            for field in (
                "checkpoint_fingerprint",
                "promotion_fingerprint",
                "rollout_manifest_fingerprint",
                "environment_fingerprint",
                "heldout_split_fingerprint",
                "taxonomy_fingerprint",
                "continuity_graph_fingerprint",
            )
        },
    }


def build_baseline_rollout_evidence(
    identity: Mapping[str, Any],
    raw_step_samples: Mapping[str, Any],
    *,
    expected_trajectory_count: int,
    expected_measured_chain_count: int,
    expected_measured_edge_count: int,
) -> dict[str, Any]:
    """Seal raw evaluate-all samples into calibration input evidence.

    Coverage is derived from per-step trajectory coordinates and the runtime
    coverage metrics.  Callers cannot provide a pre-aggregated step count or
    silently include vector-environment padding.
    """

    canonical_identity = validate_baseline_rollout_identity(identity)
    samples = _mapping(raw_step_samples, "raw continuity baseline samples")
    _exact_keys(
        samples,
        {
            "trajectory_index",
            "trajectory_step",
            "reward_imitation_total",
            "fascicle_continuity_loss",
            "fascicle_continuity_measured_chain_count",
            "fascicle_continuity_measured_edge_count",
        },
        "raw continuity baseline samples",
    )
    trajectory_index = _integer_vector(
        samples["trajectory_index"],
        "trajectory_index",
        nonnegative=True,
    )
    trajectory_step = _integer_vector(
        samples["trajectory_step"],
        "trajectory_step",
        nonnegative=True,
    )
    reward = _finite_vector(
        samples["reward_imitation_total"],
        "reward_imitation_total",
        nonnegative=True,
    )
    loss = _finite_vector(
        samples["fascicle_continuity_loss"],
        "fascicle_continuity_loss",
        nonnegative=True,
    )
    chain_count = _integer_vector(
        samples["fascicle_continuity_measured_chain_count"],
        "fascicle_continuity_measured_chain_count",
        nonnegative=True,
    )
    edge_count = _integer_vector(
        samples["fascicle_continuity_measured_edge_count"],
        "fascicle_continuity_measured_edge_count",
        nonnegative=True,
    )
    vectors = (trajectory_index, trajectory_step, reward, loss, chain_count, edge_count)
    if len({int(vector.size) for vector in vectors}) != 1:
        raise ValueError("raw continuity baseline sample lengths differ")
    if reward.size < MINIMUM_CALIBRATION_STEPS:
        raise ValueError(f"continuity calibration requires at least {MINIMUM_CALIBRATION_STEPS} held-out steps")

    expected_chains = _positive_int(
        expected_measured_chain_count,
        "expected_measured_chain_count",
    )
    expected_edges = _positive_int(
        expected_measured_edge_count,
        "expected_measured_edge_count",
    )
    expected_trajectories = _positive_int(
        expected_trajectory_count,
        "expected_trajectory_count",
    )
    _validate_step_coverage(
        trajectory_index,
        trajectory_step,
        chain_count,
        edge_count,
        expected_trajectory_count=expected_trajectories,
        expected_measured_chain_count=expected_chains,
        expected_measured_edge_count=expected_edges,
    )

    payload: dict[str, Any] = {
        "schema_version": BASELINE_ROLLOUT_SCHEMA_VERSION,
        "identity": canonical_identity,
        "coverage": {
            "trajectory_count": expected_trajectories,
            "step_count": int(reward.size),
            "measured_chain_count": expected_chains,
            "measured_edge_count": expected_edges,
        },
        "samples": {
            "trajectory_index": trajectory_index.tolist(),
            "trajectory_step": trajectory_step.tolist(),
            "imitation_reward": reward.tolist(),
            "activation_continuity_loss": loss.tolist(),
            "measured_chain_count": chain_count.tolist(),
            "measured_edge_count": edge_count.tolist(),
        },
    }
    payload["artifact_fingerprint"] = baseline_rollout_fingerprint(payload)
    return validate_baseline_rollout(payload)


def validate_baseline_rollout(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate raw per-step baseline evidence without accepting summaries."""

    _exact_keys(
        payload,
        {"schema_version", "identity", "coverage", "samples", "artifact_fingerprint"},
        "baseline rollout",
    )
    if payload["schema_version"] != BASELINE_ROLLOUT_SCHEMA_VERSION:
        raise ValueError("unsupported continuity baseline rollout schema")
    canonical_identity = validate_baseline_rollout_identity(payload["identity"])
    coverage = _mapping(payload["coverage"], "baseline coverage")
    _exact_keys(
        coverage,
        {"trajectory_count", "step_count", "measured_chain_count", "measured_edge_count"},
        "baseline coverage",
    )
    canonical_coverage = {
        key: _positive_int(coverage[key], key)
        for key in (
            "trajectory_count",
            "step_count",
            "measured_chain_count",
            "measured_edge_count",
        )
    }
    samples = _mapping(payload["samples"], "baseline samples")
    _exact_keys(
        samples,
        {
            "trajectory_index",
            "trajectory_step",
            "imitation_reward",
            "activation_continuity_loss",
            "measured_chain_count",
            "measured_edge_count",
        },
        "baseline samples",
    )
    trajectory_index = _integer_vector(
        samples["trajectory_index"],
        "trajectory_index",
        nonnegative=True,
    )
    trajectory_step = _integer_vector(
        samples["trajectory_step"],
        "trajectory_step",
        nonnegative=True,
    )
    reward = _finite_vector(
        samples["imitation_reward"],
        "imitation_reward",
        nonnegative=True,
    )
    loss = _finite_vector(
        samples["activation_continuity_loss"],
        "activation_continuity_loss",
        nonnegative=True,
    )
    chain_count = _integer_vector(
        samples["measured_chain_count"],
        "measured_chain_count",
        nonnegative=True,
    )
    edge_count = _integer_vector(
        samples["measured_edge_count"],
        "measured_edge_count",
        nonnegative=True,
    )
    vectors = (trajectory_index, trajectory_step, reward, loss, chain_count, edge_count)
    if len({int(vector.size) for vector in vectors}) != 1 or reward.size != canonical_coverage["step_count"]:
        raise ValueError("baseline sample lengths differ from coverage.step_count")
    if reward.size < MINIMUM_CALIBRATION_STEPS:
        raise ValueError(f"continuity calibration requires at least {MINIMUM_CALIBRATION_STEPS} held-out steps")
    _validate_step_coverage(
        trajectory_index,
        trajectory_step,
        chain_count,
        edge_count,
        expected_trajectory_count=canonical_coverage["trajectory_count"],
        expected_measured_chain_count=canonical_coverage["measured_chain_count"],
        expected_measured_edge_count=canonical_coverage["measured_edge_count"],
    )
    result = {
        "schema_version": BASELINE_ROLLOUT_SCHEMA_VERSION,
        "identity": canonical_identity,
        "coverage": canonical_coverage,
        "samples": {
            "trajectory_index": trajectory_index.tolist(),
            "trajectory_step": trajectory_step.tolist(),
            "imitation_reward": reward.tolist(),
            "activation_continuity_loss": loss.tolist(),
            "measured_chain_count": chain_count.tolist(),
            "measured_edge_count": edge_count.tolist(),
        },
    }
    supplied = _sha256(payload["artifact_fingerprint"], "artifact_fingerprint")
    result["artifact_fingerprint"] = supplied
    if baseline_rollout_fingerprint(result) != supplied:
        raise ValueError("continuity baseline rollout artifact fingerprint is stale")
    return result


def _validate_step_coverage(
    trajectory_index: np.ndarray,
    trajectory_step: np.ndarray,
    chain_count: np.ndarray,
    edge_count: np.ndarray,
    *,
    expected_trajectory_count: int,
    expected_measured_chain_count: int,
    expected_measured_edge_count: int,
) -> None:
    if not np.all(chain_count == expected_measured_chain_count):
        raise ValueError("raw rollout measured-chain coverage differs from the bound graph")
    if not np.all(edge_count == expected_measured_edge_count):
        raise ValueError("raw rollout measured-edge coverage differs from the bound graph")
    unique_trajectories = np.unique(trajectory_index)
    if unique_trajectories.size != expected_trajectory_count:
        raise ValueError("raw rollout trajectory coverage differs from the bound held-out split")
    if not np.array_equal(
        unique_trajectories,
        np.arange(expected_trajectory_count, dtype=np.int64),
    ):
        raise ValueError("evaluate-all trajectory indices must cover contiguous indices 0..N-1")
    coordinates = np.stack([trajectory_index, trajectory_step], axis=1)
    if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
        raise ValueError("raw continuity baseline contains duplicate trajectory steps")
    for trajectory in unique_trajectories.tolist():
        steps = trajectory_step[trajectory_index == trajectory]
        if not np.array_equal(steps, np.arange(steps.size, dtype=np.int64)):
            raise ValueError(f"trajectory {trajectory} raw steps are not one ordered prefix from frame zero")


def build_continuity_reward_calibration(
    baseline_rollout: Mapping[str, Any],
    *,
    rollout_manifest: Mapping[str, Any],
    selected_budget_fraction: float,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Compute the preregistered coefficient table and select one fixed value."""

    baseline = validate_baseline_rollout_against_manifest(
        baseline_rollout,
        rollout_manifest,
    )
    selected = _budget_fraction(selected_budget_fraction)
    floor = _positive_finite(epsilon, "epsilon")
    reward = np.asarray(baseline["samples"]["imitation_reward"], dtype=np.float64)
    loss = np.asarray(baseline["samples"]["activation_continuity_loss"], dtype=np.float64)
    median_reward = float(np.quantile(reward, 0.5))
    q95_loss = float(np.quantile(loss, 0.95))
    if median_reward <= 0.0:
        raise ValueError("baseline median imitation reward must be positive")
    if q95_loss <= 0.0:
        raise ValueError("baseline q95 activation continuity loss must be positive")
    denominator = max(q95_loss, floor)
    budgets = [
        {
            "fraction_of_median_imitation_reward": fraction,
            "absolute_penalty_budget": fraction * median_reward,
            "coefficient": fraction * median_reward / denominator,
        }
        for fraction in PENALTY_BUDGET_FRACTIONS
    ]
    selected_entry = next(entry for entry in budgets if entry["fraction_of_median_imitation_reward"] == selected)
    payload: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "source_baseline_rollout_fingerprint": baseline["artifact_fingerprint"],
        "identity": copy.deepcopy(baseline["identity"]),
        "coverage": copy.deepcopy(baseline["coverage"]),
        "statistics": {
            "median_imitation_reward": median_reward,
            "q95_activation_continuity_loss": q95_loss,
            "epsilon": floor,
            "coefficient_denominator": denominator,
        },
        "candidate_budgets": budgets,
        "selection": {
            **copy.deepcopy(selected_entry),
            "selection_is_fixed_before_training": True,
            "dynamic_validation_reweighting_allowed": False,
        },
        "formula": "lambda_cont=(budget_fraction*median_imitation_reward)/max(q95_continuity_loss,epsilon)",
    }
    payload["calibration_fingerprint"] = calibration_fingerprint(payload)
    return validate_continuity_reward_calibration(payload)


def validate_baseline_rollout_against_manifest(
    baseline_rollout: Mapping[str, Any],
    rollout_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind raw calibration evidence to its separately sealed rollout manifest."""

    from analysis.physiology_synergy.collect_continuity_baseline import (
        validate_rollout_manifest,
    )

    baseline = validate_baseline_rollout(baseline_rollout)
    manifest = validate_rollout_manifest(rollout_manifest)
    identity = baseline["identity"]
    policy = manifest["policy"]
    physiology = manifest["physiology"]
    environment = manifest["environment_manifest"]
    heldout = manifest["heldout_split_manifest"]
    expected_identity = {
        "run_id": policy["run_id"],
        "action_mode": policy["action_mode"],
        "config_hash": policy["config_hash"],
        "checkpoint_fingerprint": policy["checkpoint_fingerprint"],
        "promotion_fingerprint": policy["promotion_fingerprint"],
        "rollout_manifest_fingerprint": manifest["rollout_manifest_fingerprint"],
        "environment_fingerprint": environment["environment_fingerprint"],
        "heldout_split_fingerprint": heldout["heldout_split_fingerprint"],
        "taxonomy_fingerprint": physiology["taxonomy_fingerprint"],
        "continuity_graph_fingerprint": physiology["continuity_graph_fingerprint"],
    }
    if identity != expected_identity:
        raise ValueError("continuity baseline identity differs from its rollout manifest")
    expected_coverage = {
        "trajectory_count": heldout["trajectory_count"],
        "measured_chain_count": physiology["measured_chain_count"],
        "measured_edge_count": physiology["measured_edge_count"],
    }
    for field, expected in expected_coverage.items():
        if baseline["coverage"][field] != expected:
            raise ValueError(f"continuity baseline {field} differs from its rollout manifest")
    return baseline


def validate_continuity_reward_calibration(payload: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(
        payload,
        {
            "schema_version",
            "source_baseline_rollout_fingerprint",
            "identity",
            "coverage",
            "statistics",
            "candidate_budgets",
            "selection",
            "formula",
            "calibration_fingerprint",
        },
        "continuity calibration",
    )
    if payload["schema_version"] != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported continuity calibration schema")
    _sha256(
        payload["source_baseline_rollout_fingerprint"],
        "source_baseline_rollout_fingerprint",
    )
    identity = _mapping(payload["identity"], "calibration identity")
    # Reuse the baseline identity parser by validating the exact identity keys.
    expected_identity_fields = {
        "run_id",
        "action_mode",
        "config_hash",
        "checkpoint_fingerprint",
        "promotion_fingerprint",
        "rollout_manifest_fingerprint",
        "environment_fingerprint",
        "heldout_split_fingerprint",
        "taxonomy_fingerprint",
        "continuity_graph_fingerprint",
    }
    _exact_keys(identity, expected_identity_fields, "calibration identity")
    _action_mode(identity["action_mode"])
    _text(identity["run_id"], "run_id")
    _text(identity["config_hash"], "config_hash")
    for field in expected_identity_fields - {"run_id", "action_mode", "config_hash"}:
        _sha256(identity[field], field)
    coverage = _mapping(payload["coverage"], "calibration coverage")
    _exact_keys(
        coverage,
        {"trajectory_count", "step_count", "measured_chain_count", "measured_edge_count"},
        "calibration coverage",
    )
    for field, value in coverage.items():
        _positive_int(value, field)
    statistics = _mapping(payload["statistics"], "calibration statistics")
    _exact_keys(
        statistics,
        {
            "median_imitation_reward",
            "q95_activation_continuity_loss",
            "epsilon",
            "coefficient_denominator",
        },
        "calibration statistics",
    )
    median_reward = _positive_finite(statistics["median_imitation_reward"], "median reward")
    q95_loss = _positive_finite(statistics["q95_activation_continuity_loss"], "q95 loss")
    epsilon = _positive_finite(statistics["epsilon"], "epsilon")
    denominator = _positive_finite(statistics["coefficient_denominator"], "denominator")
    if denominator != max(q95_loss, epsilon):
        raise ValueError("calibration coefficient denominator is stale")
    budgets = payload["candidate_budgets"]
    if not isinstance(budgets, list) or len(budgets) != len(PENALTY_BUDGET_FRACTIONS):
        raise ValueError("calibration candidate budget table is incomplete")
    expected_budgets = [
        {
            "fraction_of_median_imitation_reward": fraction,
            "absolute_penalty_budget": fraction * median_reward,
            "coefficient": fraction * median_reward / denominator,
        }
        for fraction in PENALTY_BUDGET_FRACTIONS
    ]
    if budgets != expected_budgets:
        raise ValueError("calibration candidate budget table is stale")
    selection = _mapping(payload["selection"], "calibration selection")
    _exact_keys(
        selection,
        {
            "fraction_of_median_imitation_reward",
            "absolute_penalty_budget",
            "coefficient",
            "selection_is_fixed_before_training",
            "dynamic_validation_reweighting_allowed",
        },
        "calibration selection",
    )
    selected_fraction = _budget_fraction(selection["fraction_of_median_imitation_reward"])
    expected_selection = next(
        entry for entry in expected_budgets if entry["fraction_of_median_imitation_reward"] == selected_fraction
    )
    if any(selection[field] != value for field, value in expected_selection.items()):
        raise ValueError("calibration selected coefficient is stale")
    if selection["selection_is_fixed_before_training"] is not True:
        raise ValueError("continuity coefficient selection must be fixed before training")
    if selection["dynamic_validation_reweighting_allowed"] is not False:
        raise ValueError("continuity calibration cannot authorize dynamic validation reweighting")
    supplied = _sha256(payload["calibration_fingerprint"], "calibration_fingerprint")
    result = copy.deepcopy(dict(payload))
    if calibration_fingerprint(result) != supplied:
        raise ValueError("continuity calibration fingerprint is stale")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-rollout-json", type=Path, required=True)
    parser.add_argument("--rollout-manifest-json", type=Path, required=True)
    parser.add_argument(
        "--selected-budget-fraction",
        type=float,
        required=True,
        choices=PENALTY_BUDGET_FRACTIONS,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    baseline = _load_json(args.baseline_rollout_json)
    calibration = build_continuity_reward_calibration(
        baseline,
        rollout_manifest=_load_json(args.rollout_manifest_json),
        selected_budget_fraction=args.selected_budget_fraction,
    )
    _atomic_write_json(args.output_json, calibration)
    print(args.output_json.resolve())


def _action_mode(value: Any) -> str:
    result = _text(value, "action_mode")
    if result not in {"full_354", "fixed_synergy", "fixed_synergy_residual"}:
        raise ValueError("unsupported calibration action_mode")
    return result


def _budget_fraction(value: Any) -> float:
    result = _positive_finite(value, "selected_budget_fraction")
    if result not in PENALTY_BUDGET_FRACTIONS:
        raise ValueError(f"selected budget fraction must be one of {PENALTY_BUDGET_FRACTIONS}")
    return result


def _finite_vector(value: Any, field: str, *, nonnegative: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be a non-empty finite vector")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{field} must be non-negative")
    return array


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{field} fields differ from contract")


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0 or result != float(value):
        raise ValueError(f"{field} must be a positive integer")
    return result


def _integer_vector(value: Any, field: str, *, nonnegative: bool = False) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{field} must be a non-empty vector")
    if not np.issubdtype(result.dtype, np.integer):
        try:
            numeric = result.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain integers") from exc
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{field} must contain integers")
    result = result.astype(np.int64, copy=False)
    if nonnegative and np.any(result < 0):
        raise ValueError(f"{field} must be non-negative")
    return result


def _positive_finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and positive")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, field: str) -> str:
    result = _text(value, field)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{field} must be lowercase SHA-256")
    return result


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
    )
    return _mapping(payload, str(path))


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
