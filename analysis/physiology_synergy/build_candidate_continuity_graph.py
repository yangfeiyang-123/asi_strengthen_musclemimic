"""Build a non-overwriting continuity candidate graph from review v2."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from analysis.physiology_synergy.review_continuity_topology import (
    validate_topology_review,
)
from musclemimic.physiology.anatomical_groups import load_anatomical_taxonomy
from musclemimic.physiology.continuity_groups import (
    CONTINUITY_CANDIDATE_GRAPH_SCHEMA_VERSION,
    continuity_graph_fingerprint,
    load_fascicle_continuity_graph,
    validate_candidate_continuity_graph,
    validate_fascicle_continuity_graph,
)


def build_candidate_continuity_graph(
    *,
    taxonomy_path: str | Path,
    source_graph_path: str | Path,
    topology_review: Mapping[str, Any],
) -> dict[str, Any]:
    taxonomy = load_anatomical_taxonomy(taxonomy_path)
    source = load_fascicle_continuity_graph(
        source_graph_path,
        taxonomy=taxonomy,
    )
    if source.training_enabled_chain_count:
        raise ValueError("candidate source graph must remain diagnostics-only")
    review = validate_topology_review(
        topology_review,
        source_graph=source,
        taxonomy=taxonomy,
    )
    payload = source.to_manifest()
    reviewed_by_id = {entry["chain_id"]: entry for entry in review["chains"]}
    approved_ids: list[str] = []
    for chain in payload["chains"]:
        entry = reviewed_by_id[chain["chain_id"]]
        if not entry["approve_as_training_candidate"]:
            continue
        chain["edge_weights"] = copy.deepcopy(entry["approved_edge_weights"])
        chain["deadband"] = entry["approved_deadband"]
        chain["chain_weight"] = entry["approved_chain_weight"]
        chain["activity_off"] = entry["approved_activity_off"]
        chain["activity_on"] = entry["approved_activity_on"]
        chain["review_status"] = "verified_candidate"
        chain["training_enabled"] = True
        chain["provenance"] = [
            *chain["provenance"],
            *entry["provenance"],
            {
                "kind": "independent_topology_review",
                "reference": review["review_fingerprint"],
            },
        ]
        chain["notes"] = (
            str(chain.get("notes", "")).rstrip()
            + " Approved as a calibration candidate; this is not a hard-line equality claim."
        ).strip()
        approved_ids.append(chain["chain_id"])
    if not approved_ids:
        raise ValueError("topology review produced an empty candidate graph")

    payload["graph_id"] = f"{source.graph_id}_candidate_{review['review_fingerprint'][:12]}"
    generation = dict(payload.get("generation") or {})
    generation["candidate_graph"] = {
        "schema_version": CONTINUITY_CANDIDATE_GRAPH_SCHEMA_VERSION,
        "source_graph_id": source.graph_id,
        "source_graph_fingerprint": source.graph_fingerprint,
        "taxonomy_fingerprint": taxonomy.fingerprint,
        "topology_review_fingerprint": review["review_fingerprint"],
        "approved_chain_ids": approved_ids,
    }
    payload["generation"] = generation
    payload["notes"] = (
        str(payload.get("notes", "")).rstrip()
        + " Candidate parameters are frozen before baseline calibration; non-approved chains remain diagnostics-only."
    ).strip()
    payload["graph_fingerprint"] = continuity_graph_fingerprint(payload)
    candidate = validate_fascicle_continuity_graph(payload, taxonomy=taxonomy)
    validate_candidate_continuity_graph(
        candidate,
        taxonomy,
        expected_review_fingerprint=review["review_fingerprint"],
        source_graph=source,
    )
    if candidate.graph_fingerprint == source.graph_fingerprint:
        raise RuntimeError("candidate graph did not acquire a new fingerprint")
    return candidate.to_manifest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy-json", type=Path, required=True)
    parser.add_argument("--source-graph-json", type=Path, required=True)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--output-graph-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source = args.source_graph_json.resolve()
    output = args.output_graph_json.resolve()
    if source == output:
        raise ValueError("candidate output must not overwrite the topology graph")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate graph: {output}")
    candidate = build_candidate_continuity_graph(
        taxonomy_path=args.taxonomy_json,
        source_graph_path=args.source_graph_json,
        topology_review=_load_json(args.review_json),
    )
    _atomic_write_json(output, candidate)
    print(output)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(payload)


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
