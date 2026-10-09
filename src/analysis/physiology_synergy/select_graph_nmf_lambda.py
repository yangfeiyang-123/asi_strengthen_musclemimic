"""Select Graph-NMF lambda from preregistered, artifact-backed gates only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.synergy.basis_artifact import load_synergy_basis
from musclemimic.synergy.basis_factor_contract import (
    assert_matched_basis_factor_contracts,
    validate_basis_factor_contract,
)
from musclemimic.synergy.graph_nmf import (
    GRAPH_NMF_LAMBDA_SELECTION_SCHEMA_VERSION,
    GRAPH_REGULARIZATION_SCHEMA_VERSION,
    graph_nmf_lambda_selection_fingerprint,
    validate_graph_nmf_lambda_selection,
    validate_graph_regularization_manifest,
)

GRAPH_NMF_LAMBDA_CANDIDATE_INVENTORY_SCHEMA_VERSION = "graph_nmf_lambda_candidate_inventory_v1"
_SELECTION_RULE = "smallest_positive_lambda_passing_all_offline_and_dynamic_gates"


def candidate_inventory_fingerprint(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop("inventory_fingerprint", None)
    return _json_sha256(payload)


def validate_candidate_inventory(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "selection_id",
        "preregistered_lambdas",
        "candidates",
        "inventory_fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("Graph-NMF lambda candidate inventory fields differ from contract")
    if value["schema_version"] != GRAPH_NMF_LAMBDA_CANDIDATE_INVENTORY_SCHEMA_VERSION:
        raise ValueError("unsupported Graph-NMF lambda candidate inventory schema")
    preregistered = _positive_lambdas(value["preregistered_lambdas"])
    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) != len(preregistered):
        raise ValueError("Graph-NMF candidate inventory must cover every preregistered lambda exactly once")
    validated_candidates: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping) or set(item) != {
            "lambda",
            "basis_artifact_path",
            "expected_basis_artifact_fingerprint",
        }:
            raise ValueError("Graph-NMF candidate descriptor fields differ from contract")
        validated_candidates.append(
            {
                "lambda": _positive_float(item["lambda"], "candidate lambda"),
                "basis_artifact_path": _text(item["basis_artifact_path"], "candidate basis path"),
                "expected_basis_artifact_fingerprint": _sha256(
                    item["expected_basis_artifact_fingerprint"],
                    "candidate basis artifact fingerprint",
                ),
            }
        )
    if [item["lambda"] for item in validated_candidates] != preregistered:
        raise ValueError("Graph-NMF candidate descriptors must use preregistered lambda order")
    result = {
        "schema_version": GRAPH_NMF_LAMBDA_CANDIDATE_INVENTORY_SCHEMA_VERSION,
        "selection_id": _text(value["selection_id"], "selection_id"),
        "preregistered_lambdas": preregistered,
        "candidates": validated_candidates,
        "inventory_fingerprint": _sha256(value["inventory_fingerprint"], "inventory fingerprint"),
    }
    if candidate_inventory_fingerprint(result) != result["inventory_fingerprint"]:
        raise ValueError("Graph-NMF lambda candidate inventory fingerprint is stale")
    return result


def build_graph_nmf_lambda_selection(
    inventory: Mapping[str, Any],
    *,
    inventory_path: str | Path | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Load candidate basis artifacts, replay gates, and choose minimum lambda."""

    validated = validate_candidate_inventory(inventory)
    root = Path(inventory_path).expanduser().resolve().parent if inventory_path is not None else Path.cwd()
    candidates: list[tuple[float, Any, dict[str, Any], dict[str, Any]]] = []
    reference_factor: dict[str, Any] | None = None
    release_fingerprint = None
    graph_fingerprint = None
    eligible_lambdas: list[float] = []
    for descriptor in validated["candidates"]:
        path = Path(descriptor["basis_artifact_path"]).expanduser()
        if not path.is_absolute():
            path = root / path
        artifact = load_synergy_basis(path.resolve(strict=True))
        if artifact.fingerprint != descriptor["expected_basis_artifact_fingerprint"]:
            raise ValueError("Graph-NMF candidate basis fingerprint differs from inventory")
        if artifact.manifest.get("basis_family") != "graph_nmf":
            raise ValueError("Graph-NMF lambda selection rejects a non-graph basis")
        if artifact.manifest.get("basis_artifact_role") != "graph_lambda_candidate":
            raise ValueError("Graph-NMF lambda selection accepts only explicit candidate artifacts")
        factor = validate_basis_factor_contract(artifact.manifest.get("basis_factor_contract"))
        if artifact.manifest.get("basis_factor_contract_fingerprint") != factor["basis_factor_contract_fingerprint"]:
            raise ValueError("Graph-NMF candidate factor fingerprint differs from its contract")
        if reference_factor is None:
            reference_factor = factor
        else:
            assert_matched_basis_factor_contracts(reference_factor, factor)
        graph = validate_graph_regularization_manifest(artifact.manifest.get("graph_regularization"))
        if graph["schema_version"] != GRAPH_REGULARIZATION_SCHEMA_VERSION:
            raise ValueError("Graph-NMF lambda candidates require graph regularization v2")
        if graph["basis_factor_contract_fingerprint"] != factor["basis_factor_contract_fingerprint"]:
            raise ValueError("Graph-NMF graph lineage differs from candidate factor contract")
        if graph["lambda_selection_fingerprint"] is not None:
            raise ValueError("Graph-NMF lambda candidate was already bound to a selection")
        if graph["continuity_release_fingerprint"] is None:
            raise ValueError("Graph-NMF lambda candidate lacks a continuity release binding")
        coefficient = descriptor["lambda"]
        if graph["requested_lambda"] != coefficient:
            raise ValueError("Graph-NMF candidate lambda differs from artifact lineage")
        current_release = graph["continuity_release_fingerprint"]
        current_graph = graph["continuity_graph_fingerprint"]
        if release_fingerprint is None:
            release_fingerprint = current_release
            graph_fingerprint = current_graph
        elif current_release != release_fingerprint or current_graph != graph_fingerprint:
            raise ValueError("Graph-NMF lambda candidates do not share release and graph lineage")
        passed, metrics = _candidate_passes(artifact.manifest)
        if passed:
            eligible_lambdas.append(coefficient)
        candidates.append((coefficient, artifact, factor, metrics))
    if not eligible_lambdas:
        raise ValueError("no preregistered Graph-NMF lambda passes every offline and dynamic gate")
    selected_lambda = min(eligible_lambdas)
    selected_artifact = next(item[1] for item in candidates if item[0] == selected_lambda)
    assert reference_factor is not None and release_fingerprint is not None and graph_fingerprint is not None
    payload = {
        "schema_version": GRAPH_NMF_LAMBDA_SELECTION_SCHEMA_VERSION,
        "selection_id": validated["selection_id"],
        "candidate_inventory_fingerprint": validated["inventory_fingerprint"],
        "preregistered_lambdas": validated["preregistered_lambdas"],
        "eligible_lambdas": eligible_lambdas,
        "selected_lambda": selected_lambda,
        "selection_rule": _SELECTION_RULE,
        "basis_factor_contract_fingerprint": reference_factor["basis_factor_contract_fingerprint"],
        "continuity_release_fingerprint": release_fingerprint,
        "candidate_graph_fingerprint": graph_fingerprint,
        "selected_candidate_basis_fingerprint": selected_artifact.fingerprint,
        "created_at_utc": created_at_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    payload["selection_fingerprint"] = graph_nmf_lambda_selection_fingerprint(payload)
    return validate_graph_nmf_lambda_selection(payload)


def _candidate_passes(manifest: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    metrics = manifest.get("lambda_selection_metrics")
    required = {
        "all_synergy_gates_passed",
        "dynamic_coverage_required",
        "dynamic_coverage_passed",
        "heldout_global_vaf",
        "heldout_local_vaf_quantile",
        "initialization_stability",
        "split_half_stability",
        "bootstrap_stability",
        "cross_trial_stability",
        "basis_condition_number",
        "effective_rank_fraction",
        "graph_roughness",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != required:
        raise ValueError("Graph-NMF candidate lacks complete lambda_selection_metrics")
    if type(metrics["all_synergy_gates_passed"]) is not bool:
        raise ValueError("Graph-NMF all_synergy_gates_passed must be boolean")
    if type(metrics["dynamic_coverage_required"]) is not bool:
        raise ValueError("Graph-NMF dynamic_coverage_required must be boolean")
    dynamic_passed = metrics["dynamic_coverage_passed"]
    if dynamic_passed is not None and type(dynamic_passed) is not bool:
        raise ValueError("Graph-NMF dynamic_coverage_passed must be boolean or null")
    numeric = {
        key: _finite_float(metrics[key], key)
        for key in required
        if key
        not in {
            "all_synergy_gates_passed",
            "dynamic_coverage_required",
            "dynamic_coverage_passed",
        }
    }
    nonnegative = {
        "initialization_stability",
        "split_half_stability",
        "bootstrap_stability",
        "cross_trial_stability",
        "basis_condition_number",
        "effective_rank_fraction",
        "graph_roughness",
    }
    if any(numeric[key] < 0.0 for key in nonnegative):
        raise ValueError("Graph-NMF stability, conditioning, rank, and roughness must be non-negative")
    passed = metrics["all_synergy_gates_passed"] and (
        not metrics["dynamic_coverage_required"] or dynamic_passed is True
    )
    return bool(passed), {**dict(metrics), **numeric}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-inventory-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite Graph-NMF lambda selection: {args.output_json}")
    inventory = load_json_strict(args.candidate_inventory_json)
    selection = build_graph_nmf_lambda_selection(
        inventory,
        inventory_path=args.candidate_inventory_json,
    )
    _atomic_write_json(args.output_json, selection)
    print(args.output_json.resolve())


def _positive_lambdas(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError("preregistered_lambdas must be a non-empty list")
    result = [_positive_float(item, "preregistered lambda") for item in value]
    if result != sorted(set(result)):
        raise ValueError("preregistered_lambdas must be sorted and unique")
    return result


def _positive_float(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _sha256(value: Any, field: str) -> str:
    text = _text(value, field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return text


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
