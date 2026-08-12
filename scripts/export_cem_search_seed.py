#!/usr/bin/env python3
"""Export an immutable, explicitly unqualified seed from CEM state or a snapshot."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
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


def export_search_seed(
    *,
    state_path: str | Path,
    contract_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    state = Path(state_path).expanduser().resolve()
    contract_file = Path(contract_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not state.is_file():
        raise FileNotFoundError(f"CEM state is missing: {state}")
    if not contract_file.is_file():
        raise FileNotFoundError(f"CEM contract is missing: {contract_file}")
    if output.parent != contract_file.parent:
        raise ValueError("search seed must be written beside its cem_contract.json")
    if contract_file.name != "cem_contract.json":
        raise ValueError("contract path must name cem_contract.json")

    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("CEM contract must be a JSON object")
    recorded_contract_sha = contract.get("contract_sha256")
    unhashed_contract = dict(contract)
    unhashed_contract.pop("contract_sha256", None)
    if recorded_contract_sha != _json_hash(unhashed_contract):
        raise ValueError("CEM contract hash mismatch")

    state_bytes = state.read_bytes()
    state_sha = hashlib.sha256(state_bytes).hexdigest()
    with np.load(io.BytesIO(state_bytes), allow_pickle=False) as payload:
        required = {"contract_sha256", "iteration", "mean"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError("CEM state is missing fields: " + ", ".join(missing))
        if str(np.asarray(payload["contract_sha256"]).item()) != recorded_contract_sha:
            raise ValueError("CEM state is detached from its contract")
        iteration = int(np.asarray(payload["iteration"]).item())
        parameters = np.asarray(payload["mean"], dtype=np.float32)

    parameter_count = int(contract.get("parameter_count", -1))
    if (
        iteration <= 0
        or parameters.shape != (parameter_count,)
        or not np.isfinite(parameters).all()
    ):
        raise ValueError("CEM state mean has an incompatible iteration, shape, or value")
    parameter_sha = hashlib.sha256(parameters.tobytes(order="C")).hexdigest()
    seed = {
        "schema_version": "stage3_cem_search_seed_v1",
        "qualified_teacher": False,
        "seed_role": "unqualified_optimizer_mean",
        "contract_sha256": recorded_contract_sha,
        "source_state_path": str(state),
        "source_state_sha256": state_sha,
        "source_iteration": iteration,
        "parameter_f32_sha256": parameter_sha,
        "parameters": parameters.tolist(),
    }
    serialized = json.dumps(seed, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to overwrite a different search seed: {output}")
        return seed
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, output)
    return seed


def export_snapshot_candidate_seed(
    *,
    snapshot_path: str | Path,
    contract_path: str | Path,
    output_path: str | Path,
    candidate_index: int,
) -> dict[str, Any]:
    """Export one exactly indexed snapshot candidate without a teacher claim."""

    snapshot = Path(snapshot_path).expanduser().resolve()
    contract_file = Path(contract_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not snapshot.is_file():
        raise FileNotFoundError(f"CEM iteration snapshot is missing: {snapshot}")
    if not contract_file.is_file():
        raise FileNotFoundError(f"CEM contract is missing: {contract_file}")
    if output.parent != contract_file.parent:
        raise ValueError("search seed must be written beside its cem_contract.json")
    if contract_file.name != "cem_contract.json":
        raise ValueError("contract path must name cem_contract.json")
    if isinstance(candidate_index, bool) or int(candidate_index) < 0:
        raise ValueError("candidate index must be a non-negative integer")

    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("CEM contract must be a JSON object")
    recorded_contract_sha = contract.get("contract_sha256")
    unhashed_contract = dict(contract)
    unhashed_contract.pop("contract_sha256", None)
    if recorded_contract_sha != _json_hash(unhashed_contract):
        raise ValueError("CEM contract hash mismatch")

    snapshot_bytes = snapshot.read_bytes()
    with np.load(io.BytesIO(snapshot_bytes), allow_pickle=False) as payload:
        required = {"contract_sha256", "iteration", "candidates"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(
                "CEM iteration snapshot is missing fields: " + ", ".join(missing)
            )
        if str(np.asarray(payload["contract_sha256"]).item()) != recorded_contract_sha:
            raise ValueError("CEM iteration snapshot is detached from its contract")
        iteration = int(np.asarray(payload["iteration"]).item())
        candidates = np.asarray(payload["candidates"], dtype=np.float32)

    parameter_count = int(contract.get("parameter_count", -1))
    if (
        iteration <= 0
        or candidates.ndim != 2
        or candidates.shape[1] != parameter_count
        or not np.isfinite(candidates).all()
    ):
        raise ValueError("CEM snapshot candidates have an incompatible shape or value")
    index = int(candidate_index)
    if index >= candidates.shape[0]:
        raise ValueError("candidate index lies outside the snapshot population")
    parameters = np.asarray(candidates[index], dtype=np.float32)
    parameter_sha = hashlib.sha256(parameters.tobytes(order="C")).hexdigest()
    seed = {
        "schema_version": "stage3_cem_search_seed_v1",
        "qualified_teacher": False,
        "seed_role": "unqualified_snapshot_candidate",
        "contract_sha256": recorded_contract_sha,
        "source_snapshot_path": str(snapshot),
        "source_snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "source_iteration": iteration,
        "source_candidate_index": index,
        "parameter_f32_sha256": parameter_sha,
        "parameters": parameters.tolist(),
    }
    serialized = json.dumps(seed, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to overwrite a different search seed: {output}")
        return seed
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, output)
    return seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--state")
    source.add_argument("--snapshot")
    parser.add_argument("--candidate-index", type=int, default=None)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.snapshot is None:
        if args.candidate_index is not None:
            parser.error("--candidate-index requires --snapshot")
        result = export_search_seed(
            state_path=args.state,
            contract_path=args.contract,
            output_path=args.output,
        )
    else:
        if args.candidate_index is None:
            parser.error("--snapshot requires --candidate-index")
        result = export_snapshot_candidate_seed(
            snapshot_path=args.snapshot,
            contract_path=args.contract,
            output_path=args.output,
            candidate_index=args.candidate_index,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
