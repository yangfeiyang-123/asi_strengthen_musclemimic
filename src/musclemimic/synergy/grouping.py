"""Explicit muscle-region grouping with complete ordered-schema validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from musclemimic.badminton.json_contract import load_json_strict

if TYPE_CHECKING:
    from musclemimic.physiology.anatomical_groups import AnatomicalTaxonomy


def global_group(muscle_names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    names = tuple(str(name) for name in muscle_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("muscle_names must be non-empty and unique")
    return {"whole_body": tuple(range(len(names)))}


def explicit_groups(
    muscle_names: Sequence[str],
    mapping: Mapping[str, Sequence[str]],
    *,
    require_complete: bool = True,
) -> dict[str, tuple[int, ...]]:
    names = tuple(str(name) for name in muscle_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("muscle_names must be non-empty and unique")
    result: dict[str, tuple[int, ...]] = {}
    owner: dict[str, str] = {}
    for region, members in mapping.items():
        label = str(region).strip()
        member_names = tuple(str(name) for name in members)
        if not label or not member_names or len(set(member_names)) != len(member_names):
            raise ValueError("every region must have a non-empty name and unique members")
        missing = [name for name in member_names if name not in names]
        if missing:
            raise ValueError(f"region {label!r} contains unknown muscles {missing}")
        overlap = [name for name in member_names if name in owner]
        if overlap:
            raise ValueError(f"regional muscle ownership overlaps: {overlap}")
        for name in member_names:
            owner[name] = label
        result[label] = tuple(names.index(name) for name in member_names)
    unassigned = [name for name in names if name not in owner]
    if require_complete and unassigned:
        raise ValueError(f"regional mapping leaves muscles unassigned: {unassigned}")
    return result


def load_grouping_json(
    path: str | Path,
    *,
    muscle_names: Sequence[str],
    require_complete: bool = True,
    taxonomy: AnatomicalTaxonomy | None = None,
) -> dict[str, tuple[int, ...]]:
    """Load a regional grouping, optionally binding it to an anatomy taxonomy.

    Without ``taxonomy`` the ordered-schema check below is self-consistent: it only
    proves the file agrees with the *dataset* muscle order it is being applied to.
    Passing the taxonomy additionally proves that order is the compiled model's
    ordered actuator schema, which is what every intra-muscle group index means.
    """

    payload = load_json_strict(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("grouping JSON must contain an object")
    if taxonomy is not None:
        from musclemimic.physiology.synergy_binding import (
            assert_synergy_artifact_matches_taxonomy,
        )

        assert_synergy_artifact_matches_taxonomy(
            payload,
            taxonomy=taxonomy,
            context=f"grouping_json:{Path(path).name}",
        )
    mapping = payload.get("regions", payload)
    if isinstance(mapping, list):
        supplied_hash = str(payload.get("ordered_muscle_schema_sha256", ""))
        expected_hash = ordered_muscle_schema_hash(muscle_names)
        if supplied_hash != expected_hash:
            raise ValueError("indexed grouping ordered_muscle_schema_sha256 differs from the dataset schema")
        return _explicit_index_groups(
            muscle_names,
            mapping,
            require_complete=require_complete,
        )
    if not isinstance(mapping, dict):
        raise ValueError("grouping JSON must be an object or contain a regions object")
    return explicit_groups(muscle_names, mapping, require_complete=require_complete)


def ordered_muscle_schema_hash(muscle_names: Sequence[str]) -> str:
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


def _explicit_index_groups(
    muscle_names: Sequence[str],
    descriptors: Sequence[object],
    *,
    require_complete: bool,
) -> dict[str, tuple[int, ...]]:
    names = tuple(str(name) for name in muscle_names)
    result: dict[str, tuple[int, ...]] = {}
    owned: dict[int, str] = {}
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise ValueError("indexed grouping region descriptors must be objects")
        label = str(descriptor.get("name", "")).strip()
        ranges = descriptor.get("index_ranges")
        boundaries = descriptor.get("boundary_names")
        if not label or label in result or not isinstance(ranges, list) or not ranges:
            raise ValueError("indexed grouping regions require unique names and index_ranges")
        if not isinstance(boundaries, list) or len(boundaries) != len(ranges):
            raise ValueError("every indexed range requires explicit boundary_names")
        indices: list[int] = []
        for raw_range, raw_boundary in zip(ranges, boundaries, strict=True):
            if not isinstance(raw_range, list) or len(raw_range) != 2:
                raise ValueError("index_ranges entries must be [start,stop]")
            start, stop = (int(value) for value in raw_range)
            if start < 0 or stop <= start or stop > len(names):
                raise ValueError(f"indexed grouping range [{start},{stop}) is invalid")
            if (
                not isinstance(raw_boundary, list)
                or len(raw_boundary) != 2
                or [str(value) for value in raw_boundary] != [names[start], names[stop - 1]]
            ):
                raise ValueError(f"indexed grouping boundary names differ for [{start},{stop})")
            indices.extend(range(start, stop))
        if len(indices) != len(set(indices)):
            raise ValueError(f"region {label!r} contains overlapping index ranges")
        overlap = [names[index] for index in indices if index in owned]
        if overlap:
            raise ValueError(f"regional muscle ownership overlaps: {overlap}")
        for index in indices:
            owned[index] = label
        result[label] = tuple(indices)
    unassigned = [name for index, name in enumerate(names) if index not in owned]
    if require_complete and unassigned:
        raise ValueError(f"regional mapping leaves muscles unassigned: {unassigned}")
    return result
