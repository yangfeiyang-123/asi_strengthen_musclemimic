"""Promote reviewed Batch-A continuity chains into a new graph artifact."""

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

from analysis.physiology_synergy.calibrate_continuity_reward import (
    validate_continuity_reward_calibration,
)
from musclemimic.physiology.anatomical_groups import load_anatomical_taxonomy
from musclemimic.physiology.continuity_groups import (
    continuity_graph_fingerprint,
    load_fascicle_continuity_graph,
    validate_fascicle_continuity_graph,
)

CHAIN_REVIEW_SCHEMA_VERSION = "fascicle_continuity_chain_review_v1"
BATCH_A_CHAIN_IDS = (
    "right_external_oblique_continuity",
    "right_internal_oblique_continuity",
    "left_external_oblique_continuity",
    "left_internal_oblique_continuity",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def chain_review_fingerprint(payload: Mapping[str, Any]) -> str:
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("review_fingerprint", None)
    return _json_sha256(unsigned)


def validate_chain_review(payload: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(
        payload,
        {
            "schema_version",
            "batch",
            "source_graph_fingerprint",
            "taxonomy_fingerprint",
            "calibration_fingerprint",
            "reviewer",
            "chains",
            "review_fingerprint",
        },
        "chain review",
    )
    if payload["schema_version"] != CHAIN_REVIEW_SCHEMA_VERSION or payload["batch"] != "A":
        raise ValueError("the first promotion contract accepts only Batch A reviews")
    reviewer = _mapping(payload["reviewer"], "reviewer")
    _exact_keys(
        reviewer,
        {"name", "affiliation_or_role", "reviewed_at_utc", "independent_of_code_author"},
        "reviewer",
    )
    for field in ("name", "affiliation_or_role"):
        _text(reviewer[field], f"reviewer.{field}")
    timestamp = _text(reviewer["reviewed_at_utc"], "reviewer.reviewed_at_utc")
    try:
        reviewed_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("reviewed_at_utc must be an ISO-8601 timestamp") from error
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() != UTC.utcoffset(reviewed_at):
        raise ValueError("reviewed_at_utc must include an explicit UTC offset")
    if reviewer["independent_of_code_author"] is not True:
        raise ValueError("training promotion requires independent review")
    chains = payload["chains"]
    if not isinstance(chains, list) or [item.get("chain_id") for item in chains] != list(BATCH_A_CHAIN_IDS):
        raise ValueError("Batch A review must contain the four EO/IO chains in canonical order")
    canonical_chains = []
    for item in chains:
        _exact_keys(
            item,
            {"chain_id", "checks", "approved_deadband", "provenance"},
            "chain review entry",
        )
        checks = _mapping(item["checks"], f"{item['chain_id']}.checks")
        expected_checks = {
            "exact_asset_topology_reviewed",
            "same_side_verified",
            "adjacent_level_definition_reviewed",
            "baseline_activation_distribution_reviewed",
            "deadband_data_supported",
            "approve_training",
        }
        _exact_keys(checks, expected_checks, f"{item['chain_id']}.checks")
        if any(checks[field] is not True for field in expected_checks):
            raise ValueError(f"chain {item['chain_id']!r} has an incomplete promotion review")
        deadband = _nonnegative_finite(item["approved_deadband"], "approved_deadband")
        provenance = _provenance(item["provenance"], f"{item['chain_id']}.provenance")
        canonical_chains.append(
            {
                "chain_id": str(item["chain_id"]),
                "checks": dict.fromkeys(sorted(expected_checks), True),
                "approved_deadband": deadband,
                "provenance": provenance,
            }
        )
    result = {
        "schema_version": CHAIN_REVIEW_SCHEMA_VERSION,
        "batch": "A",
        "source_graph_fingerprint": _sha256(
            payload["source_graph_fingerprint"],
            "source_graph_fingerprint",
        ),
        "taxonomy_fingerprint": _sha256(
            payload["taxonomy_fingerprint"],
            "taxonomy_fingerprint",
        ),
        "calibration_fingerprint": _sha256(
            payload["calibration_fingerprint"],
            "calibration_fingerprint",
        ),
        "reviewer": {
            "name": _text(reviewer["name"], "reviewer.name"),
            "affiliation_or_role": _text(
                reviewer["affiliation_or_role"],
                "reviewer.affiliation_or_role",
            ),
            "reviewed_at_utc": timestamp,
            "independent_of_code_author": True,
        },
        "chains": canonical_chains,
    }
    supplied = _sha256(payload["review_fingerprint"], "review_fingerprint")
    result["review_fingerprint"] = supplied
    if chain_review_fingerprint(result) != supplied:
        raise ValueError("chain review fingerprint is stale")
    return result


def promote_batch_a_continuity_graph(
    *,
    taxonomy_path: str | Path,
    source_graph_path: str | Path,
    calibration: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a newly fingerprinted graph; never mutate the source artifact."""

    taxonomy = load_anatomical_taxonomy(taxonomy_path)
    graph = load_fascicle_continuity_graph(source_graph_path, taxonomy=taxonomy)
    calibrated = validate_continuity_reward_calibration(calibration)
    reviewed = validate_chain_review(review)
    if reviewed["source_graph_fingerprint"] != graph.graph_fingerprint:
        raise ValueError("chain review source graph differs from the supplied graph")
    if reviewed["taxonomy_fingerprint"] != taxonomy.fingerprint:
        raise ValueError("chain review taxonomy differs from the supplied taxonomy")
    if reviewed["calibration_fingerprint"] != calibrated["calibration_fingerprint"]:
        raise ValueError("chain review calibration fingerprint differs from supplied evidence")
    identity = calibrated["identity"]
    if identity["taxonomy_fingerprint"] != taxonomy.fingerprint:
        raise ValueError("calibration taxonomy differs from the supplied taxonomy")
    if identity["continuity_graph_fingerprint"] != graph.graph_fingerprint:
        raise ValueError("calibration graph differs from the source promotion graph")
    if calibrated["coverage"]["measured_chain_count"] != len(graph.chains):
        raise ValueError("calibration measured-chain coverage differs from the source graph")
    if calibrated["coverage"]["measured_edge_count"] != graph.edge_count:
        raise ValueError("calibration measured-edge coverage differs from the source graph")

    payload = graph.to_manifest()
    by_id = {item["chain_id"]: item for item in reviewed["chains"]}
    for chain in payload["chains"]:
        review_entry = by_id.get(chain["chain_id"])
        if review_entry is None:
            continue
        chain["deadband"] = review_entry["approved_deadband"]
        chain["review_status"] = "verified"
        chain["training_enabled"] = True
        chain["provenance"] = [
            *chain["provenance"],
            *review_entry["provenance"],
            {
                "kind": "baseline_reward_calibration",
                "reference": calibrated["calibration_fingerprint"],
            },
            {
                "kind": "independent_chain_review",
                "reference": reviewed["review_fingerprint"],
            },
        ]
        chain["notes"] = (
            str(chain.get("notes", "")).rstrip()
            + " Promoted for Batch-A matched ablation only; no hard-line equality claim."
        ).strip()
    generation = dict(payload.get("generation") or {})
    generation["training_promotion"] = {
        "batch": "A",
        "source_graph_fingerprint": graph.graph_fingerprint,
        "calibration_fingerprint": calibrated["calibration_fingerprint"],
        "review_fingerprint": reviewed["review_fingerprint"],
        "selected_reward_coefficient": calibrated["selection"]["coefficient"],
        "promoted_chain_ids": list(BATCH_A_CHAIN_IDS),
    }
    payload["generation"] = generation
    payload["notes"] = (
        str(payload.get("notes", "")).rstrip()
        + " Batch A contains only independently reviewed EO/IO chains; all other chains remain diagnostics-only."
    ).strip()
    payload["graph_fingerprint"] = continuity_graph_fingerprint(payload)
    promoted = validate_fascicle_continuity_graph(payload, taxonomy=taxonomy)
    if promoted.graph_fingerprint == graph.graph_fingerprint:
        raise RuntimeError("promoted continuity graph did not acquire a new fingerprint")
    if promoted.training_enabled_chain_count != len(BATCH_A_CHAIN_IDS):
        raise RuntimeError("promoted continuity graph has unexpected training chain coverage")
    return promoted.to_manifest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy-json", type=Path, required=True)
    parser.add_argument("--source-graph-json", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--output-graph-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source = args.source_graph_json.resolve()
    output = args.output_graph_json.resolve()
    if source == output:
        raise ValueError("promotion output must not overwrite the provisional source graph")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing promoted graph: {output}")
    promoted = promote_batch_a_continuity_graph(
        taxonomy_path=args.taxonomy_json,
        source_graph_path=args.source_graph_json,
        calibration=_load_json(args.calibration_json),
        review=_load_json(args.review_json),
    )
    _atomic_write_json(output, promoted)
    print(output)


def _provenance(value: Any, field: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
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


def _nonnegative_finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and non-negative")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
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
