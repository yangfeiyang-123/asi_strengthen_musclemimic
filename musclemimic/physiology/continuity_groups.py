"""Versioned, taxonomy-bound local fascicle continuity graphs.

Continuity chains encode adjacency between neighbouring model fascicles.  They
are independent from hard-line equality groups, soft anatomical compartments,
surface-EMG observation aggregates and functional NMF regions.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
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

FASCICLE_CONTINUITY_V1_SCHEMA_VERSION = "fascicle_continuity_graph_v1"
FASCICLE_CONTINUITY_SCHEMA_VERSION = "fascicle_continuity_graph_v2"
CONTINUITY_LOSS_SPEC_SCHEMA_VERSION = "fascicle_continuity_loss_spec_v1"
CONTINUITY_LOSS_METHOD = "robust_fascicle_continuity_v1"
CONTINUITY_LOSS_REDUCTION = "activity_gated_weighted_mean_over_chains"
CONTINUITY_LOSS_NORMALIZATION = "edge_weighted_mean_per_chain_then_activity_gated_chain_weighted_mean"
CONTINUITY_LOSS_EPS = 1e-8
CONTINUITY_CANDIDATE_GRAPH_SCHEMA_VERSION = "fascicle_continuity_candidate_graph_v1"
_SUPPORTED_CONTINUITY_SCHEMA_VERSIONS = frozenset(
    {FASCICLE_CONTINUITY_V1_SCHEMA_VERSION, FASCICLE_CONTINUITY_SCHEMA_VERSION}
)
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
    schema_version: str
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
            "schema_version": self.schema_version,
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


@dataclass(frozen=True)
class ContinuityLossSpecIdentity:
    """Canonical, non-JIT identity of one exact continuity loss."""

    schema_version: str
    graph_id: str
    graph_fingerprint: str
    taxonomy_id: str
    taxonomy_fingerprint: str
    actuator_schema_hash: str
    muscle_channel_core_fingerprint: str
    signal: str
    method: str
    scale: float
    huber_delta: float
    eps: float
    reduction: str
    normalization: str
    training_enabled_only: bool
    chain_ids: tuple[str, ...]
    chains: tuple[dict[str, Any], ...]
    loss_spec_fingerprint: str

    @property
    def chain_count(self) -> int:
        return len(self.chains)

    @property
    def edge_count(self) -> int:
        return sum(len(chain["edges"]) for chain in self.chains)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_id": self.graph_id,
            "graph_fingerprint": self.graph_fingerprint,
            "taxonomy_id": self.taxonomy_id,
            "taxonomy_fingerprint": self.taxonomy_fingerprint,
            "actuator_schema_hash": self.actuator_schema_hash,
            "muscle_channel_core_fingerprint": (self.muscle_channel_core_fingerprint),
            "signal": self.signal,
            "method": self.method,
            "scale": self.scale,
            "huber_delta": self.huber_delta,
            "eps": self.eps,
            "reduction": self.reduction,
            "normalization": self.normalization,
            "training_enabled_only": self.training_enabled_only,
            "chain_ids": list(self.chain_ids),
            "chains": copy.deepcopy(list(self.chains)),
            "chain_count": self.chain_count,
            "edge_count": self.edge_count,
            "loss_spec_fingerprint": self.loss_spec_fingerprint,
        }


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


def continuity_loss_spec_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash the complete numerical semantics of a compiled continuity loss."""

    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("loss_spec_fingerprint", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_continuity_loss_spec_identity(
    path: str | Path,
) -> ContinuityLossSpecIdentity:
    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
    )
    return validate_continuity_loss_spec_identity(payload)


def validate_continuity_loss_spec_identity(
    payload: Mapping[str, Any],
) -> ContinuityLossSpecIdentity:
    if not isinstance(payload, Mapping):
        raise ValueError("continuity loss spec identity must be a JSON object")
    _require_keys(
        payload,
        required={
            "schema_version",
            "graph_id",
            "graph_fingerprint",
            "taxonomy_id",
            "taxonomy_fingerprint",
            "actuator_schema_hash",
            "muscle_channel_core_fingerprint",
            "signal",
            "method",
            "scale",
            "huber_delta",
            "eps",
            "reduction",
            "normalization",
            "training_enabled_only",
            "chain_ids",
            "chains",
            "chain_count",
            "edge_count",
            "loss_spec_fingerprint",
        },
        context="continuity loss spec identity",
    )
    if payload["schema_version"] != CONTINUITY_LOSS_SPEC_SCHEMA_VERSION:
        raise ValueError(f"continuity loss spec schema_version must be {CONTINUITY_LOSS_SPEC_SCHEMA_VERSION!r}")
    supplied = _sha256(
        payload["loss_spec_fingerprint"],
        "loss_spec_fingerprint",
    )
    if supplied != continuity_loss_spec_fingerprint(payload):
        raise ValueError("continuity loss spec fingerprint is stale")
    signal = _nonempty_text(payload["signal"], "continuity loss spec signal")
    if signal != "activation":
        raise ValueError("continuity loss spec signal must be activation")
    method = _nonempty_text(payload["method"], "continuity loss spec method")
    if method != CONTINUITY_LOSS_METHOD:
        raise ValueError("continuity loss spec method is unsupported")
    reduction = _nonempty_text(
        payload["reduction"],
        "continuity loss spec reduction",
    )
    if reduction != CONTINUITY_LOSS_REDUCTION:
        raise ValueError("continuity loss spec reduction is unsupported")
    normalization = _nonempty_text(
        payload["normalization"],
        "continuity loss spec normalization",
    )
    if normalization != CONTINUITY_LOSS_NORMALIZATION:
        raise ValueError("continuity loss spec normalization is unsupported")
    training_only = payload["training_enabled_only"]
    if not isinstance(training_only, bool):
        raise ValueError("training_enabled_only must be boolean")
    scale = _finite_float(payload["scale"], "continuity loss spec scale")
    huber_delta = _finite_float(
        payload["huber_delta"],
        "continuity loss spec huber_delta",
    )
    eps = _finite_float(payload["eps"], "continuity loss spec eps")
    if scale <= 0.0 or huber_delta <= 0.0 or eps <= 0.0:
        raise ValueError("continuity loss spec scale, huber_delta and eps must be positive")
    chains = _validate_loss_spec_chains(payload["chains"])
    chain_ids_raw = payload["chain_ids"]
    if not isinstance(chain_ids_raw, list):
        raise ValueError("continuity loss spec chain_ids must be a list")
    chain_ids = tuple(_nonempty_text(value, "continuity loss spec chain_id") for value in chain_ids_raw)
    if chain_ids != tuple(chain["chain_id"] for chain in chains):
        raise ValueError("continuity loss spec chain_ids differ from chains")
    if training_only and not chains:
        raise ValueError("training continuity loss spec cannot be empty")
    chain_count = _nonnegative_int(
        payload["chain_count"],
        "continuity loss spec chain_count",
    )
    edge_count = _nonnegative_int(
        payload["edge_count"],
        "continuity loss spec edge_count",
    )
    if chain_count != len(chains):
        raise ValueError("continuity loss spec chain_count is stale")
    if edge_count != sum(len(chain["edges"]) for chain in chains):
        raise ValueError("continuity loss spec edge_count is stale")
    return ContinuityLossSpecIdentity(
        schema_version=CONTINUITY_LOSS_SPEC_SCHEMA_VERSION,
        graph_id=_nonempty_text(payload["graph_id"], "continuity loss spec graph_id"),
        graph_fingerprint=_sha256(
            payload["graph_fingerprint"],
            "continuity loss spec graph_fingerprint",
        ),
        taxonomy_id=_nonempty_text(
            payload["taxonomy_id"],
            "continuity loss spec taxonomy_id",
        ),
        taxonomy_fingerprint=_sha256(
            payload["taxonomy_fingerprint"],
            "continuity loss spec taxonomy_fingerprint",
        ),
        actuator_schema_hash=_sha256(
            payload["actuator_schema_hash"],
            "continuity loss spec actuator_schema_hash",
        ),
        muscle_channel_core_fingerprint=_sha256(
            payload["muscle_channel_core_fingerprint"],
            "continuity loss spec muscle_channel_core_fingerprint",
        ),
        signal=signal,
        method=method,
        scale=scale,
        huber_delta=huber_delta,
        eps=eps,
        reduction=reduction,
        normalization=normalization,
        training_enabled_only=training_only,
        chain_ids=chain_ids,
        chains=tuple(chains),
        loss_spec_fingerprint=supplied,
    )


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
    schema_version = str(payload["schema_version"])
    if schema_version not in _SUPPORTED_CONTINUITY_SCHEMA_VERSIONS:
        raise ValueError(
            f"fascicle continuity graph schema_version must be one of {sorted(_SUPPORTED_CONTINUITY_SCHEMA_VERSIONS)!r}"
        )
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
        schema_version=schema_version,
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


def validate_candidate_continuity_graph(
    graph: FascicleContinuityGraph,
    taxonomy: AnatomicalTaxonomy,
    *,
    expected_review_fingerprint: str | None = None,
    source_graph: FascicleContinuityGraph | None = None,
) -> FascicleContinuityGraph:
    """Require a v2 graph produced by the reviewed candidate builder."""

    _assert_graph_taxonomy_binding(graph, taxonomy)
    if graph.schema_version != FASCICLE_CONTINUITY_SCHEMA_VERSION:
        raise ValueError("candidate continuity graphs require fascicle_continuity_graph_v2")
    if not taxonomy.release_eligible:
        raise ValueError("candidate continuity graphs require anatomical taxonomy v2")
    generation = graph.generation
    if not isinstance(generation, Mapping):
        raise ValueError("candidate continuity graph lacks generation metadata")
    candidate = generation.get("candidate_graph")
    if not isinstance(candidate, Mapping):
        raise ValueError("continuity graph is not a reviewed candidate graph")
    _require_keys(
        candidate,
        required={
            "schema_version",
            "source_graph_id",
            "source_graph_fingerprint",
            "taxonomy_fingerprint",
            "topology_review_fingerprint",
            "approved_chain_ids",
        },
        context="candidate continuity graph generation",
    )
    if candidate["schema_version"] != CONTINUITY_CANDIDATE_GRAPH_SCHEMA_VERSION:
        raise ValueError("candidate continuity graph schema_version is unsupported")
    source_graph_id = _nonempty_text(
        candidate["source_graph_id"],
        "candidate graph source_graph_id",
    )
    source_graph_fingerprint = _sha256(
        candidate["source_graph_fingerprint"],
        "candidate graph source_graph_fingerprint",
    )
    if (
        _sha256(
            candidate["taxonomy_fingerprint"],
            "candidate graph taxonomy_fingerprint",
        )
        != taxonomy.fingerprint
    ):
        raise ValueError("candidate continuity graph taxonomy differs")
    review_fingerprint = _sha256(
        candidate["topology_review_fingerprint"],
        "candidate graph topology_review_fingerprint",
    )
    if expected_review_fingerprint is not None and review_fingerprint != _sha256(
        expected_review_fingerprint, "expected_review_fingerprint"
    ):
        raise ValueError("candidate continuity graph topology review differs")
    approved_raw = candidate["approved_chain_ids"]
    if not isinstance(approved_raw, list):
        raise ValueError("candidate continuity approved_chain_ids must be a list")
    approved = tuple(_nonempty_text(value, "candidate continuity approved chain") for value in approved_raw)
    training = tuple(chain["chain_id"] for chain in graph.chains if chain["training_enabled"])
    if not training or approved != training:
        raise ValueError("candidate continuity approved chains differ from training-enabled chains")
    invalid_status = [
        chain["chain_id"]
        for chain in graph.chains
        if chain["training_enabled"] and chain["review_status"] != "verified_candidate"
    ]
    if invalid_status:
        raise ValueError(f"candidate continuity training chains require verified_candidate status: {invalid_status}")
    if source_graph is not None:
        _assert_graph_taxonomy_binding(source_graph, taxonomy)
        if source_graph_id != source_graph.graph_id:
            raise ValueError("candidate continuity source graph id differs")
        if source_graph_fingerprint != source_graph.graph_fingerprint:
            raise ValueError("candidate continuity source graph fingerprint differs")
    return graph


def build_fascicle_continuity_spec(
    graph: FascicleContinuityGraph,
    taxonomy: AnatomicalTaxonomy,
    *,
    training_enabled_only: bool = False,
) -> FascicleContinuitySpec:
    """Compile graph chains into padded arrays outside JIT."""

    _assert_graph_taxonomy_binding(graph, taxonomy)
    chains = _selected_chains(
        graph,
        training_enabled_only=training_enabled_only,
    )
    return _compile_continuity_arrays(chains, taxonomy)


def build_continuity_loss_spec(
    graph: FascicleContinuityGraph,
    taxonomy: AnatomicalTaxonomy,
    *,
    training_enabled_only: bool,
    signal: str,
    method: str,
    scale: float,
    huber_delta: float,
    eps: float = CONTINUITY_LOSS_EPS,
) -> tuple[FascicleContinuitySpec, ContinuityLossSpecIdentity]:
    """Compile JAX arrays and their exact canonical identity in one operation."""

    _assert_graph_taxonomy_binding(graph, taxonomy)
    if signal != "activation":
        raise ValueError("continuity loss spec signal must be activation")
    if method != CONTINUITY_LOSS_METHOD:
        raise ValueError(f"continuity loss method must be {CONTINUITY_LOSS_METHOD!r}")
    scale_value = _finite_float(scale, "continuity loss scale")
    huber_value = _finite_float(huber_delta, "continuity loss huber_delta")
    eps_value = _finite_float(eps, "continuity loss eps")
    if scale_value <= 0.0 or huber_value <= 0.0 or eps_value <= 0.0:
        raise ValueError("continuity loss scale, huber_delta and eps must be positive")
    if training_enabled_only and not taxonomy.release_eligible:
        raise ValueError("training continuity loss specs require anatomical_muscle_grouping_v2")
    chains = _selected_chains(
        graph,
        training_enabled_only=training_enabled_only,
    )
    if training_enabled_only and not chains:
        raise ValueError("training continuity loss spec cannot be empty")
    spec = _compile_continuity_arrays(chains, taxonomy)
    name_to_index = {row["name"]: int(row["ordered_index"]) for row in taxonomy.ordered_actuators}
    semantic_chains = [
        {
            "chain_id": chain["chain_id"],
            "members": list(chain["members"]),
            "member_indices": [name_to_index[name] for name in chain["members"]],
            "member_weights": [1.0] * len(chain["members"]),
            "edges": copy.deepcopy(chain["edges"]),
            "edge_indices": [[name_to_index[start], name_to_index[end]] for start, end in chain["edges"]],
            "edge_weights": [float(value) for value in chain["edge_weights"]],
            "chain_weight": float(chain["chain_weight"]),
            "deadband": float(chain["deadband"]),
            "activity_off": float(chain["activity_off"]),
            "activity_on": float(chain["activity_on"]),
        }
        for chain in chains
    ]
    stable_binding = taxonomy.stable_model_binding
    payload: dict[str, Any] = {
        "schema_version": CONTINUITY_LOSS_SPEC_SCHEMA_VERSION,
        "graph_id": graph.graph_id,
        "graph_fingerprint": graph.graph_fingerprint,
        "taxonomy_id": taxonomy.taxonomy_id,
        "taxonomy_fingerprint": taxonomy.fingerprint,
        "actuator_schema_hash": stable_binding["actuator_schema_hash"],
        "muscle_channel_core_fingerprint": stable_binding["muscle_channel_core_fingerprint"],
        "signal": signal,
        "method": method,
        "scale": scale_value,
        "huber_delta": huber_value,
        "eps": eps_value,
        "reduction": CONTINUITY_LOSS_REDUCTION,
        "normalization": CONTINUITY_LOSS_NORMALIZATION,
        "training_enabled_only": bool(training_enabled_only),
        "chain_ids": [chain["chain_id"] for chain in semantic_chains],
        "chains": semantic_chains,
        "chain_count": len(semantic_chains),
        "edge_count": sum(len(chain["edges"]) for chain in semantic_chains),
    }
    payload["loss_spec_fingerprint"] = continuity_loss_spec_fingerprint(payload)
    identity = validate_continuity_loss_spec_identity(payload)
    if spec.chain_ids != identity.chain_ids:
        raise RuntimeError("compiled continuity arrays and identity chain order differ")
    return spec, identity


def assert_continuity_loss_spec_matches(
    expected: ContinuityLossSpecIdentity,
    actual: ContinuityLossSpecIdentity,
) -> None:
    if expected.loss_spec_fingerprint != actual.loss_spec_fingerprint:
        raise ValueError(
            "continuity loss spec fingerprint differs: "
            f"expected={expected.loss_spec_fingerprint}, "
            f"actual={actual.loss_spec_fingerprint}"
        )
    if expected.to_manifest() != actual.to_manifest():
        raise ValueError("continuity loss spec manifest differs despite its fingerprint")


def _selected_chains(
    graph: FascicleContinuityGraph,
    *,
    training_enabled_only: bool,
) -> list[dict[str, Any]]:
    chains = list(graph.chains)
    if training_enabled_only:
        chains = [chain for chain in chains if bool(chain["training_enabled"])]
    return chains


def _compile_continuity_arrays(
    chains: Sequence[Mapping[str, Any]],
    taxonomy: AnatomicalTaxonomy,
) -> FascicleContinuitySpec:
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
        chain["chain_id"]
        for chain in training
        if chain["review_status"] not in {"verified", "verified_candidate"} or not chain["provenance"]
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
        "actuator_schema_hash": taxonomy.stable_model_binding["actuator_schema_hash"],
        "muscle_channel_core_fingerprint": taxonomy_muscle_channel_core_fingerprint(taxonomy),
    }
    for field, expected_value in expected.items():
        if result[field] != expected_value:
            raise ValueError(f"continuity taxonomy binding differs for {field}")
    if result["runtime_compatibility"] == PORTABLE_MUSCLE_CHANNEL_ABI_COMPATIBILITY:
        bound_core = taxonomy.stable_model_binding.get("muscle_channel_core_fingerprint")
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
    if review_status not in {"provisional", "verified", "verified_candidate"}:
        raise ValueError(
            f"continuity chain {chain_id!r} review_status must be provisional, verified, or verified_candidate"
        )
    provenance = _validate_provenance(value["provenance"], f"continuity chain {chain_id}.provenance")
    training_enabled = bool(value["training_enabled"])
    if training_enabled and (review_status not in {"verified", "verified_candidate"} or not provenance):
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


def _validate_loss_spec_chains(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("continuity loss spec chains must be a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("continuity loss spec chain entries must be objects")
        _require_keys(
            raw,
            required={
                "chain_id",
                "members",
                "member_indices",
                "member_weights",
                "edges",
                "edge_indices",
                "edge_weights",
                "chain_weight",
                "deadband",
                "activity_off",
                "activity_on",
            },
            context="continuity loss spec chain",
        )
        chain_id = _nonempty_text(raw["chain_id"], "continuity loss spec chain_id")
        if chain_id in seen:
            raise ValueError("continuity loss spec chain_id values must be unique")
        seen.add(chain_id)
        members_raw = raw["members"]
        if not isinstance(members_raw, list) or len(members_raw) < 2:
            raise ValueError(f"continuity loss spec chain {chain_id!r} requires members")
        members = [_nonempty_text(member, f"continuity loss spec chain {chain_id}.member") for member in members_raw]
        if len(set(members)) != len(members):
            raise ValueError(f"continuity loss spec chain {chain_id!r} repeats members")
        member_indices = _nonnegative_int_vector(
            raw["member_indices"],
            len(members),
            f"continuity loss spec chain {chain_id}.member_indices",
        )
        if len(set(member_indices)) != len(member_indices):
            raise ValueError(f"continuity loss spec chain {chain_id!r} repeats member indices")
        member_weights = _finite_vector(
            raw["member_weights"],
            len(members),
            f"continuity loss spec chain {chain_id}.member_weights",
        )
        if any(weight <= 0.0 for weight in member_weights):
            raise ValueError("continuity loss spec member weights must be positive")
        edges_raw = raw["edges"]
        edge_indices_raw = raw["edge_indices"]
        if not isinstance(edges_raw, list) or not edges_raw:
            raise ValueError(f"continuity loss spec chain {chain_id!r} requires edges")
        if not isinstance(edge_indices_raw, list) or len(edge_indices_raw) != len(edges_raw):
            raise ValueError("continuity loss spec edge_indices length differs from edges")
        name_to_index = dict(zip(members, member_indices, strict=True))
        edges: list[list[str]] = []
        edge_indices: list[list[int]] = []
        for raw_edge, raw_indices in zip(
            edges_raw,
            edge_indices_raw,
            strict=True,
        ):
            if not isinstance(raw_edge, list) or len(raw_edge) != 2:
                raise ValueError("continuity loss spec edges must contain two names")
            start = _nonempty_text(raw_edge[0], "continuity loss spec edge start")
            end = _nonempty_text(raw_edge[1], "continuity loss spec edge end")
            if start not in name_to_index or end not in name_to_index or start == end:
                raise ValueError("continuity loss spec edge endpoints are invalid")
            indices = _nonnegative_int_vector(
                raw_indices,
                2,
                "continuity loss spec edge indices",
            )
            if indices != [name_to_index[start], name_to_index[end]]:
                raise ValueError("continuity loss spec edge names and indices disagree")
            edges.append([start, end])
            edge_indices.append(indices)
        edge_weights = _finite_vector(
            raw["edge_weights"],
            len(edges),
            f"continuity loss spec chain {chain_id}.edge_weights",
        )
        if any(weight <= 0.0 for weight in edge_weights):
            raise ValueError("continuity loss spec edge weights must be positive")
        chain_weight = _finite_float(
            raw["chain_weight"],
            f"continuity loss spec chain {chain_id}.chain_weight",
        )
        deadband = _finite_float(
            raw["deadband"],
            f"continuity loss spec chain {chain_id}.deadband",
        )
        activity_off = _finite_float(
            raw["activity_off"],
            f"continuity loss spec chain {chain_id}.activity_off",
        )
        activity_on = _finite_float(
            raw["activity_on"],
            f"continuity loss spec chain {chain_id}.activity_on",
        )
        if chain_weight <= 0.0 or deadband < 0.0:
            raise ValueError("continuity loss spec chain weight/deadband is invalid")
        if not 0.0 <= activity_off < activity_on <= 1.0:
            raise ValueError("continuity loss spec activity gate is invalid")
        result.append(
            {
                "chain_id": chain_id,
                "members": members,
                "member_indices": member_indices,
                "member_weights": member_weights,
                "edges": edges,
                "edge_indices": edge_indices,
                "edge_weights": edge_weights,
                "chain_weight": chain_weight,
                "deadband": deadband,
                "activity_off": activity_off,
                "activity_on": activity_on,
            }
        )
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


def _nonnegative_int_vector(value: Any, length: int, context: str) -> list[int]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise ValueError(f"{context} must have length {length}")
    return [_nonnegative_int(item, context) for item in value]


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be finite numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} must be finite numeric")
    return result


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a non-negative integer") from exc
    if result < 0 or result != value:
        raise ValueError(f"{context} must be a non-negative integer")
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
