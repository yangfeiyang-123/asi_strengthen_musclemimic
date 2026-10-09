#!/usr/bin/env python3
"""Derive an immutable, explicitly unqualified CEM seed by named interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def _json_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _parse_delta(value: str) -> tuple[int, str, float]:
    fields = value.split(":")
    if len(fields) != 3:
        raise ValueError("delta must use KNOT:SYNERGY:VALUE syntax")
    try:
        knot = int(fields[0])
        amount = float(fields[2])
    except ValueError as exc:
        raise ValueError("delta knot and value must be numeric") from exc
    synergy = fields[1].strip()
    if not synergy or not math.isfinite(amount) or amount == 0.0:
        raise ValueError("delta synergy must be named and value must be finite/non-zero")
    return knot, synergy, amount


def derive_search_seed(
    *,
    source_path: str | Path,
    output_path: str | Path,
    deltas: tuple[str, ...],
) -> dict[str, Any]:
    source_file = Path(source_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    if not source_file.is_file():
        raise FileNotFoundError(f"source search seed is missing: {source_file}")
    contract_file = source_file.parent / "cem_contract.json"
    if not contract_file.is_file():
        raise ValueError("source search seed has no sibling cem_contract.json")
    if output_file.parent != source_file.parent:
        raise ValueError("derived search seed must remain beside the source contract")
    if not deltas:
        raise ValueError("at least one named parameter delta is required")

    source_bytes = source_file.read_bytes()
    source = json.loads(source_bytes)
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if (
        not isinstance(source, dict)
        or source.get("schema_version") != "stage3_cem_search_seed_v1"
        or source.get("qualified_teacher") is not False
    ):
        raise ValueError("source is not an explicitly unqualified CEM search seed")
    if not isinstance(contract, dict):
        raise ValueError("CEM contract must be a JSON object")
    recorded_contract_sha = contract.get("contract_sha256")
    unhashed_contract = dict(contract)
    unhashed_contract.pop("contract_sha256", None)
    if recorded_contract_sha != _json_hash(unhashed_contract):
        raise ValueError("CEM contract hash mismatch")
    if source.get("contract_sha256") != recorded_contract_sha:
        raise ValueError("source search seed is detached from its CEM contract")

    parameters = np.asarray(source.get("parameters"), dtype=np.float32)
    parameter_count = int(contract.get("parameter_count", -1))
    if parameters.shape != (parameter_count,) or not np.isfinite(parameters).all():
        raise ValueError("source search seed parameters are incompatible")
    source_parameter_sha = hashlib.sha256(parameters.tobytes(order="C")).hexdigest()
    if source.get("parameter_f32_sha256") != source_parameter_sha:
        raise ValueError("source search seed parameter hash mismatch")
    synergy_names = tuple(str(name) for name in contract.get("synergy_names", ()))
    time_knots = int(contract.get("time_knots", 0))
    if not synergy_names or time_knots * len(synergy_names) != parameter_count:
        raise ValueError("CEM contract has no compatible named synergy layout")

    result_parameters = parameters.copy()
    interventions: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for encoded in deltas:
        knot, synergy, amount = _parse_delta(encoded)
        if knot < 0 or knot >= time_knots:
            raise ValueError(f"delta knot {knot} lies outside [0, {time_knots})")
        if synergy not in synergy_names:
            raise ValueError(f"unknown anatomical synergy in delta: {synergy}")
        key = (knot, synergy)
        if key in seen:
            raise ValueError(f"duplicate intervention for knot/synergy: {knot}:{synergy}")
        seen.add(key)
        parameter_index = knot * len(synergy_names) + synergy_names.index(synergy)
        before = float(result_parameters[parameter_index])
        after = before + float(amount)
        if not math.isfinite(after) or after < -3.0 or after > 3.0:
            raise ValueError("derived parameter lies outside the CEM [-3, 3] domain")
        result_parameters[parameter_index] = np.float32(after)
        interventions.append(
            {
                "knot_index": knot,
                "synergy_name": synergy,
                "parameter_index": parameter_index,
                "delta": float(amount),
                "before": before,
                "after": float(result_parameters[parameter_index]),
            }
        )

    parameter_sha = hashlib.sha256(
        result_parameters.tobytes(order="C")
    ).hexdigest()
    derived = {
        "schema_version": "stage3_cem_search_seed_v1",
        "qualified_teacher": False,
        "seed_role": "unqualified_parameter_intervention",
        "contract_sha256": recorded_contract_sha,
        "parent_seed_path": str(source_file),
        "parent_seed_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "parent_parameter_f32_sha256": source_parameter_sha,
        "interventions": interventions,
        "parameter_f32_sha256": parameter_sha,
        "parameters": result_parameters.tolist(),
    }
    serialized = json.dumps(derived, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output_file.exists():
        if output_file.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(
                f"refusing to overwrite a different derived seed: {output_file}"
            )
        return derived
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, output_file)
    return derived


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--delta",
        action="append",
        required=True,
        help="named additive intervention in KNOT:SYNERGY:VALUE form",
    )
    args = parser.parse_args()
    result = derive_search_seed(
        source_path=args.source,
        output_path=args.output,
        deltas=tuple(args.delta),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
