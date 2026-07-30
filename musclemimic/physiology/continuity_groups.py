"""Versioned, taxonomy-bound local fascicle continuity graphs.

Continuity chains encode adjacency between neighbouring model fascicles.  They
are independent from hard-line equality groups, soft anatomical compartments,
surface-EMG observation aggregates and functional NMF regions.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from musclemimic.physiology.anatomical_groups import (
    EXACT_RUNTIME_MODEL_COMPATIBILITY,
    PORTABLE_MUSCLE_CHANNEL_ABI_COMPATIBILITY,
    AnatomicalTaxonomy,
    taxonomy_muscle_channel_core_fingerprint,
    validate_taxonomy_against_model,
)
from musclemimic.physiology.intra_muscle import FascicleContinuitySpec
from musclemimic.physiology.synergy_binding import taxonomy_ordered_muscle_schema_hash

FASCICLE_CONTINUITY_SCHEMA_VERSION = "fascicle_continuity_graph_v1"
DEFAULT_CONTINUITY_BEHAVIOR = "diagnostics_only_no_reward"
CONTINUITY_RELATIONSHIP = "adjacent_fascicle_continuity"
_RUNTIME_COMPATIBILITIES = frozenset(
    {
        EXACT_RUNTIME_MODEL_COMPATIBILITY,
        PORTABLE_MUSCLE_CHANNEL_ABI_COMPATIBILITY,
    }
)


@dataclass(frozen=True)
class FascicleContinuityGraph:
    graph_id: str
    taxonomy_binding: dict[str, Any]
    default_behavior: str
    chains: tuple[dict[str, Any], ...]
    graph_fingerprint: str
    notes: str
    generation: dict[str, Any] | None = None
    source_path: Path | None = None

    @property
    def chain_ids(self) -> tuple[str, ...]:
        return tuple(str(chain["chain_id"]) for chain in self.chains)

    @property
    def edge_count(self) -> int:
        return sum(len(chain["edges"]) for chain in self.chains)

    @property
    def training_enabled_chain_count(self) -> int:
        return sum(bool(chain["training_enabled"]) for chain in self.chains)

    def to_manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": FASCICLE_CONTINUITY_SCHEMA_VERSION,
            "graph_id": self.graph_id,
            "taxonomy_binding": copy.deepcopy(self.taxonomy_binding),
            "default_behavior": self.default_behavior,
            "chains": copy.deepcopy(list(self.chains)),
            "notes": self.notes,
        }
        if self.generation is not None:
            payload["generation"] = copy.deepcopy(self.generation)
        payload["graph_fingerprint"] = self.graph_fingerprint
        return payload


def continuity_graph_fingerprint(payload: Mapping[str, Any]) -> str:
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("graph_fingerprint", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_fascicle_continuity_graph(
    path: str | Path,
    *,
    taxonomy: AnatomicalTaxonomy,
) -> FascicleContinuityGraph:
    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
    )
    return validate_fascicle_continuity_graph(payload, taxonomy=taxonomy, source_path=source)


def validate_fascicle_continuity_graph(
    payload: Mapping[str, Any],
    *,
    taxonomy: AnatomicalTaxonomy,
    source_path: Path | None = None,
) -> FascicleContinuityGraph:
    """Validate graph identity, taxonomy binding and every undirected edge."""

    if not isinstance(payload, Mapping):
        raise ValueError("fascicle continuity graph must be a JSON object")
    _require_keys(
        payload,
        required={
            "schema_version",
            "graph_id",
            "taxonomy_binding",
            "default_behavior",
            "chains",
            "graph_fingerprint",
            "notes",
        },
        optional={"generation"},
        context="fascicle continuity graph",
    )
    if payload["schema_version"] != FASCICLE_CONTINUITY_SCHEMA_VERSION:
        raise ValueError(f"fascicle continuity graph schema_version must be {FASCICLE_CONTINUITY_SCHEMA_VERSION!r}")
    supplied_fingerprint = _sha256(payload["graph_fingerprint"], "graph_fingerprint")
    if supplied_fingerprint != continuity_graph_fingerprint(payload):
        raise ValueError("fascicle continuity graph fingerprint is stale")
    if payload["default_behavior"] != DEFAULT_CONTINUITY_BEHAVIOR:
        raise ValueError(f"continuity default_behavior must be {DEFAULT_CONTINUITY_BEHAVIOR!r}")

    binding = _validate_taxonomy_binding(payload["taxonomy_binding"], taxonomy=taxonomy)
    chains_raw = payload["chains"]
    if not isinstance(chains_raw, list):
        raise ValueError("continuity chains must be a list")
    actuator_by_name = {row["name"]: row for row in taxonomy.ordered_actuators}
    chains: list[dict[str, Any]] = []
    global_edges: set[tuple[str, str]] = set()
    for raw in chains_raw:
        chain = _validate_chain(raw, actuator_by_name=actuator_by_name)
        for start, end in chain["edges"]:
            undirected = tuple(sorted((start, end)))
            if undirected in global_edges:
                raise ValueError(f"duplicate undirected continuity edge: {undirected}")
            global_edges.add(undirected)
        chains.append(chain)
    chain_ids = [chain["chain_id"] for chain in chains]
    if len(set(chain_ids)) != len(chain_ids):
        raise ValueError("continuity chain_id values must be unique")

    generation = payload.get("generation")
    if generation is not None:
        if not isinstance(generation, Mapping):
            raise ValueError("continuity generation must be an object")
        generation = copy.deepcopy(dict(generation))
    return FascicleContinuityGraph(
        graph_id=_nonempty_text(payload["graph_id"], "graph_id"),
        taxonomy_binding=binding,
        default_behavior=DEFAULT_CONTINUITY_BEHAVIOR,
        chains=tuple(chains),
        graph_fingerprint=supplied_fingerprint,
        notes=str(payload["notes"]),
        generation=generation,
        source_path=source_path,
    )


def validate_continuity_graph_against_model(
    graph: FascicleContinuityGraph,
    taxonomy: AnatomicalTaxonomy,
    model: Any,
    *,
    validate_package_version: bool = True,
) -> None:
    """Bind a validated graph to either the exact scene or portable muscle ABI."""

    _assert_graph_taxonomy_binding(graph, taxonomy)
    validate_taxonomy_against_model(
        taxonomy,
        model,
        compatibility=graph.taxonomy_binding["runtime_compatibility"],
        validate_package_version=validate_package_version,
    )


def build_fascicle_continuity_spec(
    graph: FascicleContinuityGraph,
    taxonomy: AnatomicalTaxonomy,
    *,
    training_enabled_only: bool = False,
) -> FascicleContinuitySpec:
    """Compile graph chains into padded arrays outside JIT."""

    _assert_graph_taxonomy_binding(graph, taxonomy)
    chains = list(graph.chains)
    if training_enabled_only:
        chains = [chain for chain in chains if bool(chain["training_enabled"])]
    max_edges = max((len(chain["edges"]) for chain in chains), default=1)
    max_members = max((len(chain["members"]) for chain in chains), default=1)
    edge_indices = np.zeros((len(chains), max_edges, 2), dtype=np.int32)
    edge_mask = np.zeros((len(chains), max_edges), dtype=np.float32)
    edge_weights = np.zeros((len(chains), max_edges), dtype=np.float32)
    member_indices = np.zeros((len(chains), max_members), dtype=np.int32)
    member_mask = np.zeros((len(chains), max_members), dtype=np.float32)
    member_weights = np.zeros((len(chains), max_members), dtype=np.float32)
    name_to_index = {row["name"]: int(row["ordered_index"]) for row in taxonomy.ordered_actuators}
    for chain_index, chain in enumerate(chains):
        edge_count = len(chain["edges"])
        member_count = len(chain["members"])
        edge_indices[chain_index, :edge_count] = [
            [name_to_index[start], name_to_index[end]] for start, end in chain["edges"]
        ]
        edge_mask[chain_index, :edge_count] = 1.0
        edge_weights[chain_index, :edge_count] = np.asarray(chain["edge_weights"], dtype=np.float32)
        member_indices[chain_index, :member_count] = [name_to_index[name] for name in chain["members"]]
        member_mask[chain_index, :member_count] = 1.0
        member_weights[chain_index, :member_count] = 1.0
    return FascicleContinuitySpec(
        edge_indices=jnp.asarray(edge_indices),
        edge_mask=jnp.asarray(edge_mask),
        edge_weights=jnp.asarray(edge_weights),
        member_indices=jnp.asarray(member_indices),
        member_mask=jnp.asarray(member_mask),
        member_weights=jnp.asarray(member_weights),
        chain_weights=jnp.asarray([chain["chain_weight"] for chain in chains], dtype=jnp.float32),
        deadband=jnp.asarray([chain["deadband"] for chain in chains], dtype=jnp.float32),
        activity_off=jnp.asarray([chain["activity_off"] for chain in chains], dtype=jnp.float32),
        activity_on=jnp.asarray([chain["activity_on"] for chain in chains], dtype=jnp.float32),
        activation_addresses=jnp.asarray(
            [row["actadr"] for row in taxonomy.ordered_actuators],
            dtype=jnp.int32,
        ),
        body_actuator_ids=jnp.asarray(
            [row["actuator_id"] for row in taxonomy.ordered_actuators],
            dtype=jnp.int32,
        ),
        chain_ids=tuple(chain["chain_id"] for chain in chains),
    )


def resolve_fascicle_continuity_reward_gate(
    graph: FascicleContinuityGraph,
    *,
    enabled: bool,
    require_verified_training_chains: bool = True,
) -> tuple[bool, str]:
    training = [chain for chain in graph.chains if bool(chain["training_enabled"])]
    if not enabled:
        return False, "fascicle_continuity_reward_disabled_by_config"
    if require_verified_training_chains and not training:
        raise ValueError("fascicle continuity reward requested but the graph has no verified training-enabled chains")
    invalid = [
        chain["chain_id"] for chain in training if chain["review_status"] != "verified" or not chain["provenance"]
    ]
    if invalid:
        raise ValueError(f"fascicle continuity reward has unverified training chains: {invalid}")
    if not training:
        raise ValueError("fascicle continuity reward cannot operate on an empty training graph")
    return True, "fascicle_continuity_reward_active_verified_chains"


def _validate_taxonomy_binding(value: Any, *, taxonomy: AnatomicalTaxonomy) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("continuity taxonomy_binding must be an object")
    _require_keys(
        value,
        required={
            "taxonomy_id",
            "taxonomy_fingerprint",
            "ordered_muscle_schema_sha256",
            "actuator_schema_hash",
            "muscle_channel_core_fingerprint",
            "runtime_compatibility",
        },
        context="continuity taxonomy_binding",
    )
    result = {
        "taxonomy_id": _nonempty_text(value["taxonomy_id"], "taxonomy_binding.taxonomy_id"),
        "taxonomy_fingerprint": _sha256(
            value["taxonomy_fingerprint"],
            "taxonomy_binding.taxonomy_fingerprint",
        ),
        "ordered_muscle_schema_sha256": _sha256(
            value["ordered_muscle_schema_sha256"],
            "taxonomy_binding.ordered_muscle_schema_sha256",
        ),
        "actuator_schema_hash": _sha256(
            value["actuator_schema_hash"],
            "taxonomy_binding.actuator_schema_hash",
        ),
        "muscle_channel_core_fingerprint": _sha256(
            value["muscle_channel_core_fingerprint"],
            "taxonomy_binding.muscle_channel_core_fingerprint",
        ),
        "runtime_compatibility": _nonempty_text(
            value["runtime_compatibility"],
            "taxonomy_binding.runtime_compatibility",
        ),
    }
    if result["runtime_compatibility"] not in _RUNTIME_COMPATIBILITIES:
        raise ValueError("continuity runtime_compatibility is unsupported")
    expected = {
        "taxonomy_id": taxonomy.taxonomy_id,
        "taxonomy_fingerprint": taxonomy.fingerprint,
        "ordered_muscle_schema_sha256": taxonomy_ordered_muscle_schema_hash(taxonomy),
        "actuator_schema_hash": taxonomy.model_binding["actuator_schema_hash"],
        "muscle_channel_core_fingerprint": taxonomy_muscle_channel_core_fingerprint(taxonomy),
    }
    for field, expected_value in expected.items():
        if result[field] != expected_value:
            raise ValueError(f"continuity taxonomy binding differs for {field}")
    if result["runtime_compatibility"] == PORTABLE_MUSCLE_CHANNEL_ABI_COMPATIBILITY:
        bound_core = taxonomy.model_binding.get("muscle_channel_core_fingerprint")
        if bound_core != result["muscle_channel_core_fingerprint"]:
            raise ValueError("portable continuity graph requires taxonomy model-binding core fingerprint")
    return result


def _validate_chain(
    value: Any,
    *,
    actuator_by_name: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("continuity chain entries must be objects")
    _require_keys(
        value,
        required={
            "chain_id",
            "side",
            "anatomical_structure",
            "members",
            "edges",
            "edge_weights",
            "deadband",
            "chain_weight",
            "activity_off",
            "activity_on",
            "review_status",
            "training_enabled",
            "provenance",
        },
        optional={"notes"},
        context="continuity chain",
    )
    chain_id = _nonempty_text(value["chain_id"], "continuity chain_id")
    side = _nonempty_text(value["side"], f"continuity chain {chain_id}.side")
    if side not in {"left", "right"}:
        raise ValueError(f"continuity chain {chain_id!r} must declare left or right side")
    members_raw = value["members"]
    if not isinstance(members_raw, list) or len(members_raw) < 2:
        raise ValueError(f"continuity chain {chain_id!r} requires at least two members")
    members = [_nonempty_text(member, f"continuity chain {chain_id}.member") for member in members_raw]
    if len(set(members)) != len(members):
        raise ValueError(f"continuity chain {chain_id!r} contains duplicate members")
    unknown = sorted(set(members) - set(actuator_by_name))
    if unknown:
        raise ValueError(f"continuity chain {chain_id!r} contains unknown members: {unknown}")
    wrong_side = [name for name in members if actuator_by_name[name]["side"] != side]
    if wrong_side:
        raise ValueError(f"continuity chain {chain_id!r} mixes or misdeclares sides: {wrong_side}")

    edges_raw = value["edges"]
    if not isinstance(edges_raw, list) or not edges_raw:
        raise ValueError(f"continuity chain {chain_id!r} requires non-empty edges")
    edges: list[list[str]] = []
    local_edges: set[tuple[str, str]] = set()
    touched: set[str] = set()
    for raw_edge in edges_raw:
        if not isinstance(raw_edge, list | tuple) or len(raw_edge) != 2:
            raise ValueError(f"continuity chain {chain_id!r} edge must contain two members")
        start = _nonempty_text(raw_edge[0], f"continuity chain {chain_id}.edge start")
        end = _nonempty_text(raw_edge[1], f"continuity chain {chain_id}.edge end")
        if start == end:
            raise ValueError(f"continuity chain {chain_id!r} contains a self-edge")
        if start not in members or end not in members:
            raise ValueError(f"continuity chain {chain_id!r} edge endpoint is outside members")
        undirected = tuple(sorted((start, end)))
        if undirected in local_edges:
            raise ValueError(f"continuity chain {chain_id!r} repeats an undirected edge")
        local_edges.add(undirected)
        touched.update((start, end))
        edges.append([start, end])
    if touched != set(members) or not _is_connected(members, edges):
        raise ValueError(f"continuity chain {chain_id!r} edges must connect every member")

    weights = _finite_vector(value["edge_weights"], len(edges), f"continuity chain {chain_id}.edge_weights")
    if any(weight <= 0.0 for weight in weights):
        raise ValueError(f"continuity chain {chain_id!r} edge weights must be positive")
    deadband = _finite_float(value["deadband"], f"continuity chain {chain_id}.deadband")
    chain_weight = _finite_float(value["chain_weight"], f"continuity chain {chain_id}.chain_weight")
    activity_off = _finite_float(value["activity_off"], f"continuity chain {chain_id}.activity_off")
    activity_on = _finite_float(value["activity_on"], f"continuity chain {chain_id}.activity_on")
    if deadband < 0.0 or chain_weight <= 0.0:
        raise ValueError(f"continuity chain {chain_id!r} deadband/weight is invalid")
    if not 0.0 <= activity_off < activity_on <= 1.0:
        raise ValueError(f"continuity chain {chain_id!r} activity gate is invalid")
    if not isinstance(value["training_enabled"], bool):
        raise ValueError(f"continuity chain {chain_id!r} training_enabled must be boolean")
    review_status = _nonempty_text(value["review_status"], f"continuity chain {chain_id}.review_status")
    if review_status not in {"provisional", "verified"}:
        raise ValueError(f"continuity chain {chain_id!r} review_status must be provisional or verified")
    provenance = _validate_provenance(value["provenance"], f"continuity chain {chain_id}.provenance")
    training_enabled = bool(value["training_enabled"])
    if training_enabled and (review_status != "verified" or not provenance):
        raise ValueError(f"training-enabled continuity chain {chain_id!r} requires verified review and provenance")
    if review_status == "provisional" and training_enabled:
        raise ValueError(f"provisional continuity chain {chain_id!r} cannot drive reward")
    result = {
        "chain_id": chain_id,
        "side": side,
        "anatomical_structure": _nonempty_text(
            value["anatomical_structure"],
            f"continuity chain {chain_id}.anatomical_structure",
        ),
        "members": members,
        "edges": edges,
        "edge_weights": weights,
        "deadband": deadband,
        "chain_weight": chain_weight,
        "activity_off": activity_off,
        "activity_on": activity_on,
        "review_status": review_status,
        "training_enabled": training_enabled,
        "provenance": provenance,
    }
    if "notes" in value:
        result["notes"] = str(value["notes"])
    return result


def _assert_graph_taxonomy_binding(
    graph: FascicleContinuityGraph,
    taxonomy: AnatomicalTaxonomy,
) -> None:
    expected = validate_fascicle_continuity_graph(graph.to_manifest(), taxonomy=taxonomy)
    if expected.graph_fingerprint != graph.graph_fingerprint:
        raise ValueError("continuity graph object changed after validation")


def _is_connected(members: list[str], edges: list[list[str]]) -> bool:
    adjacency = {member: set() for member in members}
    for start, end in edges:
        adjacency[start].add(end)
        adjacency[end].add(start)
    visited: set[str] = set()
    pending = [members[0]]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency[current] - visited)
    return visited == set(members)


def _validate_provenance(value: Any, context: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    result: list[dict[str, str]] = []
    for entry in value:
        if not isinstance(entry, Mapping) or set(entry) != {"kind", "reference"}:
            raise ValueError(f"{context} entries require exactly kind/reference")
        result.append(
            {
                "kind": _nonempty_text(entry["kind"], f"{context}.kind"),
                "reference": _nonempty_text(entry["reference"], f"{context}.reference"),
            }
        )
    return result


def _finite_vector(value: Any, length: int, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{context} must have length {length}")
    return [_finite_float(item, context) for item in value]


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be finite numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} must be finite numeric")
    return result


def _nonempty_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, context: str) -> str:
    text = _nonempty_text(value, context)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return text


def _require_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    context: str,
    optional: set[str] | None = None,
) -> None:
    actual = set(value)
    optional = set() if optional is None else optional
    missing = sorted(required - actual)
    extra = sorted(actual - required - optional)
    if missing or extra:
        raise ValueError(f"{context} keys differ: missing={missing}, extra={extra}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in continuity graph: {key!r}")
        result[key] = value
    return result
