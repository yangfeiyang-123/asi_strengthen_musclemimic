#!/usr/bin/env python3
"""Build the provisional MyoFullBody trunk fascicle continuity graph."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from musclemimic.physiology.anatomical_groups import (  # noqa: E402
    PORTABLE_MUSCLE_CHANNEL_ABI_COMPATIBILITY,
    load_anatomical_taxonomy,
    taxonomy_muscle_channel_core_fingerprint,
)
from musclemimic.physiology.continuity_groups import (  # noqa: E402
    DEFAULT_CONTINUITY_BEHAVIOR,
    FASCICLE_CONTINUITY_SCHEMA_VERSION,
    continuity_graph_fingerprint,
    validate_continuity_graph_against_model,
    validate_fascicle_continuity_graph,
)
from musclemimic.physiology.synergy_binding import taxonomy_ordered_muscle_schema_hash  # noqa: E402

GRAPH_ID = "myofullbody_354_trunk_fascicle_continuity_v2"
EXPECTED_TAXONOMY_FINGERPRINT = "c044f7d4b1d037c314cc04ef209f3dbb89e652935cf3063a30b38881fb255d27"
DEFAULT_TAXONOMY_PATH = REPOSITORY_ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_curated_v2.json"
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "configs/physiology/myofullbody_354_fascicle_continuity_v2.json"


def _chain(
    side: str,
    structure: str,
    members: list[str],
) -> dict[str, Any]:
    if len(members) < 2:
        raise ValueError(f"continuity chain {side}/{structure} resolved fewer than two members")
    edges = [[members[index], members[index + 1]] for index in range(len(members) - 1)]
    return {
        "chain_id": f"{side}_{structure}_continuity",
        "side": side,
        "anatomical_structure": structure,
        "members": members,
        "edges": edges,
        "edge_weights": [1.0] * len(edges),
        "deadband": 0.15,
        "chain_weight": 1.0,
        "activity_off": 0.02,
        "activity_on": 0.10,
        "review_status": "provisional",
        "training_enabled": False,
        "provenance": [
            {
                "kind": "model_asset_topology",
                "reference": "musclemimic-models==1.0.5:model/torso/assets/myotorso_assets.xml",
            }
        ],
        "notes": (
            "Adjacent-level project prior; diagnostics only. This is neither an "
            "Exo hard-line equivalence nor a shared neural-drive claim."
        ),
    }


def _members(side_suffix: str) -> list[tuple[str, list[str], int]]:
    return [
        ("external_oblique", [f"EO{index}_{side_suffix}" for index in range(1, 7)], 6),
        ("internal_oblique", [f"IO{index}_{side_suffix}" for index in range(1, 7)], 6),
        ("iliocostalis_lumbar", [f"IL_L{index}_{side_suffix}" for index in range(1, 5)], 4),
        ("iliocostalis_rib", [f"IL_R{index}_{side_suffix}" for index in range(5, 13)], 8),
        (
            "longissimus_thoracis_thoracic",
            [f"LTpT_T{index}_{side_suffix}" for index in range(1, 13)],
            12,
        ),
        (
            "longissimus_thoracis_rib",
            [f"LTpT_R{index}_{side_suffix}" for index in range(4, 13)],
            9,
        ),
        ("longissimus_lumbar", [f"LTpL_L{index}_{side_suffix}" for index in range(1, 6)], 5),
        ("psoas_transverse_process", [f"Ps_L{index}_TP_{side_suffix}" for index in range(1, 6)], 5),
        (
            "psoas_intervertebral_disc",
            [f"Ps_L{index}_L{index + 1}_IVD_{side_suffix}" for index in range(1, 5)],
            4,
        ),
        ("multifidus_spinous", [f"MF_m{index}s_{side_suffix}" for index in range(1, 6)], 5),
        (
            "multifidus_transverse_1",
            [f"MF_m{index}t.1_{side_suffix}" for index in range(1, 6)],
            5,
        ),
        (
            "multifidus_transverse_2",
            [f"MF_m{index}t.2_{side_suffix}" for index in range(1, 6)],
            5,
        ),
        (
            "multifidus_transverse_3",
            [f"MF_m{index}t.3_{side_suffix}" for index in range(1, 6)],
            5,
        ),
        (
            "multifidus_laminar",
            [f"MF_m{index}.laminar_{side_suffix}" for index in range(1, 6)],
            5,
        ),
    ]


def build_continuity_graph(taxonomy, *, expected_taxonomy_fingerprint: str) -> dict[str, Any]:
    if taxonomy.fingerprint != str(expected_taxonomy_fingerprint):
        raise ValueError(f"curated taxonomy fingerprint differs from the pinned graph parent: {taxonomy.fingerprint}")
    if taxonomy.hard_line_groups:
        raise ValueError("v1 continuity graph requires the curated taxonomy hard-line set to remain empty")
    known = set(taxonomy.actuator_names)
    chains: list[dict[str, Any]] = []
    for side, suffix in (("right", "r"), ("left", "l")):
        specs = _members(suffix)
        if len(specs) != 14:
            raise RuntimeError("v1 continuity inventory must contain 14 structures per side")
        for structure, members, expected_count in specs:
            if len(members) != expected_count:
                raise RuntimeError(
                    f"continuity pattern {side}/{structure} count changed: {len(members)} != {expected_count}"
                )
            missing = [name for name in members if name not in known]
            if missing:
                raise ValueError(f"continuity pattern {side}/{structure} has unresolved members: {missing}")
            chains.append(_chain(side, structure, members))
    if len(chains) != 28:
        raise RuntimeError(f"v1 continuity graph must contain 28 chains, got {len(chains)}")
    if sum(len(chain["edges"]) for chain in chains) != 140:
        raise RuntimeError("v1 continuity graph edge count must equal 140")

    payload: dict[str, Any] = {
        "schema_version": FASCICLE_CONTINUITY_SCHEMA_VERSION,
        "graph_id": GRAPH_ID,
        "taxonomy_binding": {
            "taxonomy_id": taxonomy.taxonomy_id,
            "taxonomy_fingerprint": taxonomy.fingerprint,
            "ordered_muscle_schema_sha256": taxonomy_ordered_muscle_schema_hash(taxonomy),
            "actuator_schema_hash": taxonomy.stable_model_binding["actuator_schema_hash"],
            "muscle_channel_core_fingerprint": taxonomy_muscle_channel_core_fingerprint(taxonomy),
            "runtime_compatibility": PORTABLE_MUSCLE_CHANNEL_ABI_COMPATIBILITY,
        },
        "default_behavior": DEFAULT_CONTINUITY_BEHAVIOR,
        "chains": chains,
        "generation": {
            "tool": "scripts/build_myofullbody_fascicle_continuity.py",
            "policy": "explicit_reviewed_candidate_chains_no_prefix_inference",
            "chain_count_per_side": 14,
            "training_promotion_policy": "independent_reviewed_artifact_required",
        },
        "notes": (
            "Provisional diagnostics-only adjacency graph for same-side, same-series, "
            "neighbouring trunk fascicles. No chain is training-enabled."
        ),
    }
    payload["graph_fingerprint"] = continuity_graph_fingerprint(payload)
    validate_fascicle_continuity_graph(payload, taxonomy=taxonomy)
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--expected-taxonomy-fingerprint", default=EXPECTED_TAXONOMY_FINGERPRINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--skip-runtime-model-validation",
        action="store_true",
        help="Only for source-only generation where model assets cannot be imported.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    taxonomy = load_anatomical_taxonomy(args.taxonomy)
    payload = build_continuity_graph(
        taxonomy,
        expected_taxonomy_fingerprint=args.expected_taxonomy_fingerprint,
    )
    graph = validate_fascicle_continuity_graph(payload, taxonomy=taxonomy)
    for chain in graph.chains:
        print(f"{chain['chain_id']}: {', '.join(chain['members'])}")
    if not args.skip_runtime_model_validation:
        from musclemimic.environments.humanoids.myofullbody import MyoFullBody

        validate_continuity_graph_against_model(
            graph,
            taxonomy,
            MyoFullBody(disable_fingers=True)._model,
        )
    _atomic_write_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
