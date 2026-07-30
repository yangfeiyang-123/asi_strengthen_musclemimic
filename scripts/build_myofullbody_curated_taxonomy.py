#!/usr/bin/env python3
"""Build the reviewed diagnostics-only MyoFullBody 354-muscle taxonomy."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from musclemimic.physiology.anatomical_groups import (  # noqa: E402
    AnatomicalTaxonomy,
    load_anatomical_taxonomy,
    taxonomy_fingerprint,
    taxonomy_muscle_channel_core_fingerprint,
    validate_anatomical_taxonomy,
    validate_taxonomy_against_model,
)

CURATED_TAXONOMY_ID = "myofullbody_354_muscle_taxonomy_curated_v1"
EXPECTED_AUDIT_FINGERPRINT = "084dea06ea0206dd7981b52f80d2b3e19bd8a6e004888554de75d45e087c23ea"
DEFAULT_AUDIT_PATH = REPOSITORY_ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_audit_v1.json"
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_curated_v1.json"


def _group(
    group_id: str,
    side: str,
    anatomical_muscle: str,
    members: list[str],
    *,
    notes: str,
) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "side": side,
        "anatomical_muscle": anatomical_muscle,
        "members": members,
        "relationship": "soft_compartment_group",
        "review_status": "provisional",
        "training_enabled": False,
        "member_weights": [1.0] * len(members),
        "deadband": 0.15,
        "group_weight": 1.0,
        "activity_off": 0.02,
        "activity_on": 0.10,
        "provenance": [
            {
                "kind": "model_asset_inventory",
                "reference": (
                    "musclemimic-models==1.0.5:"
                    + (
                        "model/torso/assets/myotorso_assets.xml"
                        if anatomical_muscle in {"external_oblique", "internal_oblique"}
                        else (
                            "model/arm/assets/myoarm_bimanual_assets.xml"
                            if side in {"right", "left"}
                            and any(
                                token in group_id
                                for token in ("deltoid", "pectoralis", "latissimus", "biceps", "triceps")
                            )
                            else "model/leg/assets/myolegs_assets.xml"
                        )
                    )
                ),
            }
        ],
        "notes": notes,
    }


def curated_soft_compartment_groups() -> list[dict[str, Any]]:
    """Return the exact diagnostics-only compartment inventory."""

    groups: list[dict[str, Any]] = []
    upper = (
        ("deltoid", ["DELT1", "DELT2", "DELT3"]),
        ("pectoralis_major", ["PECM1", "PECM2", "PECM3"]),
        ("latissimus_dorsi", ["LAT1", "LAT2", "LAT3"]),
        ("biceps_brachii_heads", ["BIClong", "BICshort"]),
        ("triceps_brachii_heads", ["TRIlong", "TRIlat", "TRImed"]),
    )
    for side, suffix in (("right", ""), ("left", "_left")):
        for structure, base_members in upper:
            muscle = structure.removesuffix("_heads")
            groups.append(
                _group(
                    f"{side}_{structure}_compartments",
                    side,
                    muscle,
                    [f"{name}{suffix}" for name in base_members],
                    notes=(
                        "Diagnostics only. These are anatomical compartments or heads, "
                        "not verified hard-equivalent numerical lines."
                    ),
                )
            )

    lower = (
        ("gluteus_maximus", ["glmax1", "glmax2", "glmax3"]),
        ("gluteus_medius", ["glmed1", "glmed2", "glmed3"]),
        ("gluteus_minimus", ["glmin1", "glmin2", "glmin3"]),
        ("adductor_magnus", ["addmagProx", "addmagMid", "addmagDist", "addmagIsch"]),
        ("gastrocnemius_heads", ["gasmed", "gaslat"]),
    )
    for side, suffix in (("right", "_r"), ("left", "_l")):
        for structure, base_members in lower:
            groups.append(
                _group(
                    f"{side}_{structure}_compartments",
                    side,
                    structure.removesuffix("_heads"),
                    [f"{name}{suffix}" for name in base_members],
                    notes=(
                        "Diagnostics only. Compartment/head dispersion must not be "
                        "used as hard equality or PPO supervision."
                    ),
                )
            )

    for side, suffix in (("right", "_r"), ("left", "_l")):
        for abbreviation, structure in (("EO", "external_oblique"), ("IO", "internal_oblique")):
            groups.append(
                _group(
                    f"{side}_{structure}_broad_compartments",
                    side,
                    structure,
                    [f"{abbreviation}{index}{suffix}" for index in range(1, 7)],
                    notes=(
                        "Broad mean-based description only. Long trunk series use the "
                        "independent adjacency continuity graph for local diagnostics."
                    ),
                )
            )
    return groups


def build_curated_taxonomy(
    audit: AnatomicalTaxonomy,
    *,
    expected_audit_fingerprint: str = EXPECTED_AUDIT_FINGERPRINT,
) -> dict[str, Any]:
    """Derive a curated manifest without mutating the audit inventory."""

    if audit.fingerprint != str(expected_audit_fingerprint):
        raise ValueError(f"audit taxonomy fingerprint differs from the pinned reviewed parent: {audit.fingerprint}")
    if any(
        (
            audit.hard_line_groups,
            audit.soft_compartment_groups,
            audit.observation_aggregates,
            audit.functional_synergy_regions,
        )
    ):
        raise ValueError("curation parent must be the relationship-free audit inventory")

    payload = deepcopy(audit.to_manifest())
    payload.pop("taxonomy_fingerprint", None)
    payload["taxonomy_id"] = CURATED_TAXONOMY_ID
    payload["model_binding"]["muscle_channel_core_fingerprint"] = taxonomy_muscle_channel_core_fingerprint(audit)
    payload["hard_line_groups"] = []
    payload["soft_compartment_groups"] = curated_soft_compartment_groups()
    payload["observation_aggregates"] = []
    payload["functional_synergy_regions"] = []
    payload["generation"] = {
        "tool": "scripts/build_myofullbody_curated_taxonomy.py",
        "parent_taxonomy_id": audit.taxonomy_id,
        "parent_taxonomy_fingerprint": audit.fingerprint,
        "curation_policy": "diagnostic_compartments_only_no_hard_inference",
        "curation_version": 1,
    }
    payload["notes"] = (
        "Curated diagnostics-only compartments for the 354-channel no-finger model. "
        "Hard-line groups remain empty; EMG aggregates and functional NMF regions "
        "remain independent assets."
    )
    payload["taxonomy_fingerprint"] = taxonomy_fingerprint(payload)
    validate_anatomical_taxonomy(payload)
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
    parser.add_argument("--audit-taxonomy", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--expected-audit-fingerprint", default=EXPECTED_AUDIT_FINGERPRINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--skip-runtime-model-validation",
        action="store_true",
        help="Only for source-only generation where model assets cannot be imported.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audit = load_anatomical_taxonomy(args.audit_taxonomy)
    payload = build_curated_taxonomy(
        audit,
        expected_audit_fingerprint=args.expected_audit_fingerprint,
    )
    taxonomy = validate_anatomical_taxonomy(payload)
    if not args.skip_runtime_model_validation:
        from musclemimic.environments.humanoids.myofullbody import MyoFullBody

        validate_taxonomy_against_model(taxonomy, MyoFullBody(disable_fingers=True)._model)
    _atomic_write_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
