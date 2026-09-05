"""Seal an independent, calibration-free continuity topology review."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.physiology.anatomical_groups import (
    AnatomicalTaxonomy,
    load_anatomical_taxonomy,
)
from musclemimic.physiology.continuity_groups import (
    FascicleContinuityGraph,
    load_fascicle_continuity_graph,
)

TOPOLOGY_REVIEW_SCHEMA_VERSION = "fascicle_continuity_topology_review_v2"
TOPOLOGY_REVIEW_CHECKS = (
    "exact_asset_topology_reviewed",
    "same_side_verified",
    "adjacent_level_definition_reviewed",
    "not_hard_line_equivalence",
    "baseline_activation_distribution_reviewed",
    "deadband_data_supported",
)
_STRUCTURAL_CHECKS = TOPOLOGY_REVIEW_CHECKS[:4]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def topology_review_fingerprint(payload: Mapping[str, Any]) -> str:
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("review_fingerprint", None)
    return _json_sha256(unsigned)


def seal_topology_review(
    draft: Mapping[str, Any],
    *,
    source_graph: FascicleContinuityGraph,
    taxonomy: AnatomicalTaxonomy,
) -> dict[str, Any]:
    """Validate an unsigned human review and add its canonical fingerprint."""

    if "review_fingerprint" in draft:
        raise ValueError("topology review draft must not pre-fill review_fingerprint")
    payload = copy.deepcopy(dict(draft))
    payload["review_fingerprint"] = topology_review_fingerprint(payload)
    return validate_topology_review(
        payload,
        source_graph=source_graph,
        taxonomy=taxonomy,
    )


def validate_topology_review(
    payload: Mapping[str, Any],
    *,
    source_graph: FascicleContinuityGraph | None = None,
    taxonomy: AnatomicalTaxonomy | None = None,
) -> dict[str, Any]:
    _exact_keys(
        payload,
        {
            "schema_version",
            "source_graph_fingerprint",
            "taxonomy_fingerprint",
            "reviewer",
            "chains",
            "review_fingerprint",
        },
        "topology review",
    )
    if payload["schema_version"] != TOPOLOGY_REVIEW_SCHEMA_VERSION:
        raise ValueError(f"topology review schema_version must be {TOPOLOGY_REVIEW_SCHEMA_VERSION!r}")
    source_fingerprint = _sha256(
        payload["source_graph_fingerprint"],
        "source_graph_fingerprint",
    )
    taxonomy_fingerprint = _sha256(
        payload["taxonomy_fingerprint"],
        "taxonomy_fingerprint",
    )
    if source_graph is not None:
        if source_fingerprint != source_graph.graph_fingerprint:
            raise ValueError("topology review source graph fingerprint differs")
        if source_graph.training_enabled_chain_count:
            raise ValueError("topology review source graph must remain diagnostics-only")
    if taxonomy is not None:
        if not taxonomy.release_eligible:
            raise ValueError("topology review requires anatomical taxonomy v2")
        if taxonomy_fingerprint != taxonomy.fingerprint:
            raise ValueError("topology review taxonomy fingerprint differs")

    reviewer = _mapping(payload["reviewer"], "reviewer")
    _exact_keys(
        reviewer,
        {
            "name",
            "affiliation_or_role",
            "reviewed_at_utc",
            "independent_of_code_author",
        },
        "reviewer",
    )
    timestamp = _utc_timestamp(reviewer["reviewed_at_utc"])
    if reviewer["independent_of_code_author"] is not True:
        raise ValueError("candidate topology requires independent review")
    canonical_reviewer = {
        "name": _text(reviewer["name"], "reviewer.name"),
        "affiliation_or_role": _text(
            reviewer["affiliation_or_role"],
            "reviewer.affiliation_or_role",
        ),
        "reviewed_at_utc": timestamp,
        "independent_of_code_author": True,
    }

    chains_raw = payload["chains"]
    if not isinstance(chains_raw, list) or not chains_raw:
        raise ValueError("topology review chains must be a non-empty list")
    expected_chain_ids = None
    source_by_id: dict[str, dict[str, Any]] = {}
    if source_graph is not None:
        expected_chain_ids = list(source_graph.chain_ids)
        source_by_id = {chain["chain_id"]: chain for chain in source_graph.chains}
        actual_chain_ids = [item.get("chain_id") for item in chains_raw]
        if actual_chain_ids != expected_chain_ids:
            raise ValueError("topology review must cover every source chain in canonical order")

    canonical_chains = []
    for item in chains_raw:
        item_id = item.get("chain_id") if isinstance(item, Mapping) else None
        canonical_chains.append(
            _validate_review_chain(
                item,
                source_chain=source_by_id.get(item_id),
            )
        )
    chain_ids = [item["chain_id"] for item in canonical_chains]
    if len(set(chain_ids)) != len(chain_ids):
        raise ValueError("topology review chain_id values must be unique")
    if not any(item["approve_as_training_candidate"] for item in canonical_chains):
        raise ValueError("topology review approves no training candidate chains")

    result = {
        "schema_version": TOPOLOGY_REVIEW_SCHEMA_VERSION,
        "source_graph_fingerprint": source_fingerprint,
        "taxonomy_fingerprint": taxonomy_fingerprint,
        "reviewer": canonical_reviewer,
        "chains": canonical_chains,
    }
    supplied = _sha256(payload["review_fingerprint"], "review_fingerprint")
    result["review_fingerprint"] = supplied
    if topology_review_fingerprint(result) != supplied:
        raise ValueError("topology review fingerprint is stale")
    return result


def _validate_review_chain(
    value: Any,
    *,
    source_chain: Mapping[str, Any] | None,
) -> dict[str, Any]:
    item = _mapping(value, "topology review chain")
    _exact_keys(
        item,
        {
            "chain_id",
            "approve_as_training_candidate",
            "checks",
            "approved_deadband",
            "approved_edge_weights",
            "approved_chain_weight",
            "approved_activity_off",
            "approved_activity_on",
            "provenance",
        },
        "topology review chain",
    )
    chain_id = _text(item["chain_id"], "topology review chain_id")
    approved = item["approve_as_training_candidate"]
    if not isinstance(approved, bool):
        raise ValueError("approve_as_training_candidate must be boolean")
    checks = _mapping(item["checks"], f"{chain_id}.checks")
    _exact_keys(checks, set(TOPOLOGY_REVIEW_CHECKS), f"{chain_id}.checks")
    if any(not isinstance(checks[field], bool) for field in TOPOLOGY_REVIEW_CHECKS):
        raise ValueError(f"chain {chain_id!r} review checks must be boolean")
    if any(checks[field] is not True for field in _STRUCTURAL_CHECKS):
        raise ValueError(f"chain {chain_id!r} has incomplete structural review")
    if approved and any(checks[field] is not True for field in TOPOLOGY_REVIEW_CHECKS):
        raise ValueError(f"chain {chain_id!r} has incomplete candidate review")

    edge_count = len(source_chain["edges"]) if source_chain is not None else None
    edge_weights = _positive_vector(
        item["approved_edge_weights"],
        field=f"{chain_id}.approved_edge_weights",
        expected_length=edge_count,
    )
    deadband = _nonnegative_finite(
        item["approved_deadband"],
        f"{chain_id}.approved_deadband",
    )
    chain_weight = _positive_finite(
        item["approved_chain_weight"],
        f"{chain_id}.approved_chain_weight",
    )
    activity_off = _nonnegative_finite(
        item["approved_activity_off"],
        f"{chain_id}.approved_activity_off",
    )
    activity_on = _positive_finite(
        item["approved_activity_on"],
        f"{chain_id}.approved_activity_on",
    )
    if not 0.0 <= activity_off < activity_on <= 1.0:
        raise ValueError(f"chain {chain_id!r} approved activity gate is invalid")
    provenance = _provenance(
        item["provenance"],
        f"{chain_id}.provenance",
    )
    if approved and not provenance:
        raise ValueError(f"approved chain {chain_id!r} requires review provenance")
    return {
        "chain_id": chain_id,
        "approve_as_training_candidate": approved,
        "checks": {field: bool(checks[field]) for field in TOPOLOGY_REVIEW_CHECKS},
        "approved_deadband": deadband,
        "approved_edge_weights": edge_weights,
        "approved_chain_weight": chain_weight,
        "approved_activity_off": activity_off,
        "approved_activity_on": activity_on,
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy-json", type=Path, required=True)
    parser.add_argument("--source-graph-json", type=Path, required=True)
    parser.add_argument("--review-draft-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = args.output_json.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite topology review: {output}")
    taxonomy = load_anatomical_taxonomy(args.taxonomy_json)
    graph = load_fascicle_continuity_graph(
        args.source_graph_json,
        taxonomy=taxonomy,
    )
    review = seal_topology_review(
        _load_json(args.review_draft_json),
        source_graph=graph,
        taxonomy=taxonomy,
    )
    _atomic_write_json(output, review)
    print(output)


def _provenance(value: Any, field: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result = []
    for entry in value:
        _exact_keys(entry, {"kind", "reference"}, field)
        result.append(
            {
                "kind": _text(entry["kind"], f"{field}.kind"),
                "reference": _text(entry["reference"], f"{field}.reference"),
            }
        )
    return result


def _positive_vector(
    value: Any,
    *,
    field: str,
    expected_length: int | None,
) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    if expected_length is not None and len(value) != expected_length:
        raise ValueError(f"{field} length differs from source graph edges")
    return [_positive_finite(item, field) for item in value]


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{field} fields differ from contract")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value.strip()


def _sha256(value: Any, field: str) -> str:
    result = _text(value, field)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{field} must be lowercase SHA-256")
    return result


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite numeric")
    return result


def _nonnegative_finite(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _positive_finite(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _utc_timestamp(value: Any) -> str:
    text = _text(value, "reviewer.reviewed_at_utc")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewed_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("reviewed_at_utc must include an explicit UTC offset")
    return text


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
