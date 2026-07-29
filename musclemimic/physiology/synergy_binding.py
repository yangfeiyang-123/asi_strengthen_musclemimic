"""Fail-closed binding between muscle-synergy artifacts and the anatomy taxonomy.

Both halves of the muscle stack already pin their own ordered channel schema, but
they pin it under different names:

* the synergy stack hashes its ordered muscle names into
  ``muscle_schema_sha256`` (:mod:`musclemimic.synergy.basis_artifact`) or
  ``ordered_muscle_schema_sha256`` (:mod:`musclemimic.synergy.grouping`);
* the anatomy taxonomy hashes the compiled model's ordered actuators into
  ``actuator_schema_hash`` and binds them to ``runtime_model_hash``
  (:mod:`musclemimic.physiology.anatomical_groups`).

Each hash is self-consistent on its own side, so a synergy basis fit on one
channel order and a taxonomy exported from a different model can both validate
individually while disagreeing about what channel *k* means.  Intra-muscle
consistency addresses muscles purely by position, so such a disagreement would
silently relabel every group member instead of failing: the reported dispersion
would be computed over muscles that are not the ones the group names.

The two hashes use identical canonical JSON (``ensure_ascii=False``,
``sort_keys=True``, ``separators=(",", ":")``) over ``{"muscle_names": [...]}``,
so they are directly comparable.  These validators are the missing cross-check.

:func:`ordered_muscle_schema_sha256` re-states that canonicalization here instead
of importing it from :mod:`musclemimic.synergy.grouping`.  That keeps this module
stdlib-only, so the synergy stack can call into physiology for the cross-check
without an import cycle -- and it turns the cross-side agreement into something a
test asserts (``tests/unit/test_physiology_synergy_binding.py``) rather than a
coincidence between two independently maintained ``json.dumps`` calls.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from musclemimic.physiology.anatomical_groups import AnatomicalTaxonomy

SYNERGY_TAXONOMY_BINDING_SCHEMA_VERSION = "physiology_synergy_taxonomy_binding_v1"

# Field names under which synergy artifacts publish their ordered-muscle hash.
SYNERGY_SCHEMA_HASH_FIELDS = (
    "muscle_schema_sha256",
    "ordered_muscle_schema_sha256",
)

# A synergy artifact partitions channels for NMF or rank allocation, so the only
# class it may declare is the functional one.  This is a whitelist rather than a
# blacklist of the three anatomy classes on purpose: the taxonomy's own top-level
# keys are plural (``hard_line_groups``, ``observation_aggregates``, ...), and
# ``build_intra_muscle_spec`` takes those plural names, so the single most likely
# mislabeling is a copied plural key -- which any blacklist of singular spellings
# would wave through.
_FUNCTIONAL_GROUPING_CLASS = "functional_synergy_region"


def ordered_muscle_schema_sha256(muscle_names: Sequence[str]) -> str:
    """Hash an ordered muscle-name list with the synergy stack's canonical JSON.

    Byte-compatible with ``musclemimic.synergy.grouping.ordered_muscle_schema_hash``
    and with ``basis_artifact._json_sha256({"muscle_names": [...]})``; that
    three-way agreement is asserted by test rather than assumed.
    """

    names = tuple(str(name) for name in muscle_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("muscle_names must be non-empty and unique")
    encoded = json.dumps(
        {"muscle_names": list(names)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def taxonomy_ordered_muscle_schema_hash(taxonomy: AnatomicalTaxonomy) -> str:
    """Hash the taxonomy's ordered actuator names the way the synergy stack does."""

    return ordered_muscle_schema_sha256(taxonomy.actuator_names)


def assert_taxonomy_matches_ordered_muscles(
    taxonomy: AnatomicalTaxonomy,
    muscle_names: Sequence[str],
    *,
    context: str,
) -> dict[str, Any]:
    """Require the taxonomy and a synergy channel order to be the same schema.

    Returns a binding record for the caller's report.  Raises ``ValueError`` with
    the first divergent index when the two orders disagree, because a positional
    diagnosis is what makes such a mismatch actionable.
    """

    anatomy_names = taxonomy.actuator_names
    synergy_names = tuple(str(name) for name in muscle_names)
    if not synergy_names:
        raise ValueError(f"{context}: synergy muscle_names must be non-empty")
    if len(synergy_names) != len(set(synergy_names)):
        raise ValueError(f"{context}: synergy muscle_names must be unique")
    if len(anatomy_names) != len(synergy_names):
        raise ValueError(
            f"{context}: taxonomy describes {len(anatomy_names)} ordered actuators "
            f"but the synergy schema describes {len(synergy_names)} muscles"
        )
    for index, (anatomy_name, synergy_name) in enumerate(zip(anatomy_names, synergy_names, strict=True)):
        if anatomy_name != synergy_name:
            raise ValueError(
                f"{context}: taxonomy and synergy ordered muscle schemas diverge at "
                f"index {index}: taxonomy has {anatomy_name!r}, synergy has {synergy_name!r}"
            )
    return _taxonomy_binding_record(taxonomy, context=context)


def _taxonomy_binding_record(
    taxonomy: AnatomicalTaxonomy,
    *,
    context: str,
) -> dict[str, Any]:
    """Describe the taxonomy side of a verified binding.

    ``actuator_schema_hash`` and ``runtime_model_hash`` are indexed rather than
    ``.get``-defaulted: ``_validate_model_binding`` makes both mandatory, so an
    absent key means a hand-built taxonomy, and publishing ``""`` there would let
    the record claim a successful binding while carrying empty provenance.
    """

    return {
        "schema_version": SYNERGY_TAXONOMY_BINDING_SCHEMA_VERSION,
        "context": context,
        "taxonomy_id": taxonomy.taxonomy_id,
        "taxonomy_fingerprint": taxonomy.fingerprint,
        "actuator_count": len(taxonomy.actuator_names),
        "ordered_muscle_schema_sha256": taxonomy_ordered_muscle_schema_hash(taxonomy),
        "actuator_schema_hash": str(taxonomy.model_binding["actuator_schema_hash"]),
        "runtime_model_hash": str(taxonomy.model_binding["runtime_model_hash"]),
    }


def assert_synergy_artifact_matches_taxonomy(
    manifest: Mapping[str, Any],
    *,
    taxonomy: AnatomicalTaxonomy,
    context: str,
) -> dict[str, Any]:
    """Bind a synergy manifest to the taxonomy through its declared schema hash.

    ``manifest`` is any synergy artifact payload that publishes one of
    :data:`SYNERGY_SCHEMA_HASH_FIELDS`.  When it also carries explicit
    ``muscle_names`` those are compared element-wise, which reports the divergent
    index rather than only an opaque hash difference.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError(f"{context}: synergy manifest must be a mapping")

    grouping_class = str(manifest.get("grouping_class", "")).strip()
    if grouping_class and grouping_class.casefold() != _FUNCTIONAL_GROUPING_CLASS:
        raise ValueError(
            f"{context}: synergy artifact declares grouping_class {grouping_class!r}; "
            f"only {_FUNCTIONAL_GROUPING_CLASS!r} may be bound here, because a "
            "functional synergy partition must never be adopted as an anatomical "
            "relationship"
        )

    declared = {
        field: str(manifest[field]).strip() for field in SYNERGY_SCHEMA_HASH_FIELDS if manifest.get(field) is not None
    }
    if not declared:
        raise ValueError(
            f"{context}: synergy artifact must publish one of "
            f"{list(SYNERGY_SCHEMA_HASH_FIELDS)} to be bound to a taxonomy"
        )

    names = manifest.get("muscle_names")
    if names is not None:
        if not isinstance(names, Sequence) or isinstance(names, str | bytes):
            raise ValueError(f"{context}: synergy muscle_names must be a sequence")
        record = assert_taxonomy_matches_ordered_muscles(
            taxonomy,
            names,
            context=context,
        )
    else:
        record = _taxonomy_binding_record(taxonomy, context=context)

    expected = record["ordered_muscle_schema_sha256"]
    for field, value in sorted(declared.items()):
        if value != expected:
            raise ValueError(
                f"{context}: synergy artifact {field}={value!r} does not match the "
                f"taxonomy ordered muscle schema hash {expected!r}"
            )

    record["verified_synergy_hash_fields"] = sorted(declared)
    if grouping_class:
        record["grouping_class"] = grouping_class
    return record
