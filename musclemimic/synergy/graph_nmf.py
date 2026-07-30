"""Verified continuity-graph bindings for optional Laplacian NMF.

The standard synergy fit remains graph-free.  A positive graph coefficient is
accepted only when a taxonomy-bound continuity graph contains reviewed,
training-enabled chains.  The resulting adjacency is immutable and every
regional/preprocessing subset receives its own content-bound manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.physiology.anatomical_groups import load_anatomical_taxonomy
from musclemimic.physiology.continuity_groups import (
    load_fascicle_continuity_graph,
    resolve_fascicle_continuity_reward_gate,
)
from musclemimic.physiology.synergy_binding import (
    assert_taxonomy_matches_ordered_muscles,
    ordered_muscle_schema_sha256,
)

GRAPH_REGULARIZATION_SCHEMA_VERSION = "synergy_graph_regularization_v1"
LAPLACIAN_NMF_METHOD = "laplacian_nmf_v1"
GRAPH_NORMALIZATION_SPACE = "source_signal_units_no_channel_normalization"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GraphRegularizationBinding:
    """One immutable weighted adjacency aligned to ordered NMF channels."""

    adjacency: np.ndarray
    muscle_names: tuple[str, ...]
    manifest: dict[str, Any]

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.muscle_names)
        adjacency = _validate_adjacency(self.adjacency, width=len(names))
        manifest = validate_graph_regularization_manifest(self.manifest)
        if manifest["ordered_muscle_schema_sha256"] != ordered_muscle_schema_sha256(names):
            raise ValueError("graph regularization ordered muscle schema is stale")
        if manifest["edge_count"] != _adjacency_edge_count(adjacency):
            raise ValueError("graph regularization edge_count differs from adjacency")
        if manifest["edge_set_fingerprint"] != _edge_set_fingerprint(adjacency, names):
            raise ValueError("graph regularization edge-set fingerprint is stale")
        adjacency.setflags(write=False)
        object.__setattr__(self, "adjacency", adjacency)
        object.__setattr__(self, "muscle_names", names)
        object.__setattr__(self, "manifest", manifest)

    @property
    def lambda_value(self) -> float:
        return float(self.manifest["lambda"])

    @property
    def edge_count(self) -> int:
        return int(self.manifest["edge_count"])

    @property
    def enabled(self) -> bool:
        return bool(self.manifest["enabled"])

    @property
    def degree(self) -> np.ndarray:
        return np.sum(self.adjacency, axis=1)

    def subset(
        self,
        indices: Sequence[int],
        *,
        scope: str,
    ) -> GraphRegularizationBinding:
        selected = np.asarray(tuple(int(index) for index in indices), dtype=np.int64)
        if selected.ndim != 1 or selected.size == 0:
            raise ValueError("graph regularization subset must be non-empty")
        if np.any(selected < 0) or np.any(selected >= len(self.muscle_names)):
            raise ValueError("graph regularization subset indices are out of range")
        if np.unique(selected).size != selected.size:
            raise ValueError("graph regularization subset indices must be unique")
        names = tuple(self.muscle_names[int(index)] for index in selected)
        adjacency = self.adjacency[np.ix_(selected, selected)].copy()
        edge_count = _adjacency_edge_count(adjacency)
        manifest = {
            **copy.deepcopy(self.manifest),
            "enabled": edge_count > 0,
            "lambda": self.manifest["requested_lambda"] if edge_count > 0 else 0.0,
            "edge_count": edge_count,
            "ordered_muscle_schema_sha256": ordered_muscle_schema_sha256(names),
            "edge_set_fingerprint": _edge_set_fingerprint(adjacency, names),
            "scope": _nonempty_text(scope, "graph regularization scope"),
        }
        return GraphRegularizationBinding(adjacency, names, manifest)

    def restrict_to(
        self,
        indices: Sequence[int],
        *,
        scope: str,
    ) -> GraphRegularizationBinding:
        """Keep the current channel ABI while dropping edges outside a subset."""

        selected = np.asarray(tuple(int(index) for index in indices), dtype=np.int64)
        if selected.ndim != 1 or selected.size == 0:
            raise ValueError("graph regularization restriction must be non-empty")
        if np.any(selected < 0) or np.any(selected >= len(self.muscle_names)):
            raise ValueError("graph regularization restriction indices are out of range")
        if np.unique(selected).size != selected.size:
            raise ValueError("graph regularization restriction indices must be unique")
        adjacency = np.zeros_like(self.adjacency)
        adjacency[np.ix_(selected, selected)] = self.adjacency[np.ix_(selected, selected)]
        edge_count = _adjacency_edge_count(adjacency)
        manifest = {
            **copy.deepcopy(self.manifest),
            "enabled": edge_count > 0,
            "lambda": self.manifest["requested_lambda"] if edge_count > 0 else 0.0,
            "edge_count": edge_count,
            "edge_set_fingerprint": _edge_set_fingerprint(adjacency, self.muscle_names),
            "scope": _nonempty_text(scope, "graph regularization scope"),
        }
        return GraphRegularizationBinding(adjacency, self.muscle_names, manifest)

    def retain_disjoint_subgraphs(
        self,
        index_groups: Sequence[Sequence[int]],
        *,
        scope: str,
    ) -> GraphRegularizationBinding:
        """Retain only edges internal to the supplied disjoint channel groups.

        Regional NMF fits never see cross-region edges.  This operation creates
        the exact full-width adjacency represented by a block-diagonal regional
        composite, rather than incorrectly claiming that the whole graph
        regularized every component.
        """

        adjacency = np.zeros_like(self.adjacency)
        seen: set[int] = set()
        for raw_group in index_groups:
            selected = np.asarray(tuple(int(index) for index in raw_group), dtype=np.int64)
            if selected.ndim != 1 or selected.size == 0:
                raise ValueError("graph regularization subgraph groups must be non-empty")
            if np.any(selected < 0) or np.any(selected >= len(self.muscle_names)):
                raise ValueError("graph regularization subgraph index is out of range")
            current = {int(index) for index in selected.tolist()}
            if len(current) != selected.size or seen & current:
                raise ValueError("graph regularization subgraph groups must be unique and disjoint")
            seen.update(current)
            adjacency[np.ix_(selected, selected)] = self.adjacency[np.ix_(selected, selected)]
        edge_count = _adjacency_edge_count(adjacency)
        manifest = {
            **copy.deepcopy(self.manifest),
            "enabled": edge_count > 0,
            "lambda": self.manifest["requested_lambda"] if edge_count > 0 else 0.0,
            "edge_count": edge_count,
            "edge_set_fingerprint": _edge_set_fingerprint(adjacency, self.muscle_names),
            "scope": _nonempty_text(scope, "graph regularization scope"),
        }
        return GraphRegularizationBinding(adjacency, self.muscle_names, manifest)


def load_verified_graph_regularization(
    *,
    taxonomy_path: str | Path,
    continuity_path: str | Path,
    expected_taxonomy_fingerprint: str,
    expected_continuity_fingerprint: str,
    muscle_names: Sequence[str],
    lambda_value: float,
) -> GraphRegularizationBinding:
    """Load verified training chains and compile their weighted adjacency."""

    coefficient = _positive_finite(lambda_value, "graph regularization lambda")
    taxonomy = load_anatomical_taxonomy(taxonomy_path)
    expected_taxonomy = _sha256(expected_taxonomy_fingerprint, "expected taxonomy fingerprint")
    if taxonomy.fingerprint != expected_taxonomy:
        raise ValueError("graph-NMF taxonomy fingerprint differs from the pinned config")
    assert_taxonomy_matches_ordered_muscles(
        taxonomy,
        muscle_names,
        context="graph-NMF full ordered channels",
    )
    graph = load_fascicle_continuity_graph(continuity_path, taxonomy=taxonomy)
    expected_graph = _sha256(expected_continuity_fingerprint, "expected continuity graph fingerprint")
    if graph.graph_fingerprint != expected_graph:
        raise ValueError("graph-NMF continuity fingerprint differs from the pinned config")
    # The same verified/provenance gate used by online reward is deliberately
    # reused here: graph-NMF must never train on provisional diagnostic edges.
    resolve_fascicle_continuity_reward_gate(
        graph,
        enabled=True,
        require_verified_training_chains=True,
    )

    names = tuple(str(name) for name in muscle_names)
    name_to_index = {name: index for index, name in enumerate(names)}
    adjacency = np.zeros((len(names), len(names)), dtype=np.float64)
    chain_ids: list[str] = []
    for chain in graph.chains:
        if not chain["training_enabled"]:
            continue
        chain_ids.append(str(chain["chain_id"]))
        chain_weight = float(chain["chain_weight"])
        for (start, end), edge_weight in zip(
            chain["edges"],
            chain["edge_weights"],
            strict=True,
        ):
            i = name_to_index[start]
            j = name_to_index[end]
            weight = chain_weight * float(edge_weight)
            adjacency[i, j] += weight
            adjacency[j, i] += weight
    edge_count = _adjacency_edge_count(adjacency)
    if edge_count <= 0:
        raise ValueError("graph-NMF verified training graph resolved zero edges")
    manifest = {
        "schema_version": GRAPH_REGULARIZATION_SCHEMA_VERSION,
        "enabled": True,
        "method": LAPLACIAN_NMF_METHOD,
        "lambda": coefficient,
        "requested_lambda": coefficient,
        "continuity_graph_id": graph.graph_id,
        "continuity_graph_fingerprint": graph.graph_fingerprint,
        "taxonomy_id": taxonomy.taxonomy_id,
        "taxonomy_fingerprint": taxonomy.fingerprint,
        "edge_count": edge_count,
        "full_graph_edge_count": edge_count,
        "normalization_space": GRAPH_NORMALIZATION_SPACE,
        "training_enabled_only": True,
        "chain_ids": chain_ids,
        "ordered_muscle_schema_sha256": ordered_muscle_schema_sha256(names),
        "edge_set_fingerprint": _edge_set_fingerprint(adjacency, names),
        "scope": "whole_body",
    }
    return GraphRegularizationBinding(adjacency, names, manifest)


def validate_graph_regularization_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the lineage object embedded in basis/candidate artifacts."""

    required = {
        "schema_version",
        "enabled",
        "method",
        "lambda",
        "requested_lambda",
        "continuity_graph_id",
        "continuity_graph_fingerprint",
        "taxonomy_id",
        "taxonomy_fingerprint",
        "edge_count",
        "full_graph_edge_count",
        "normalization_space",
        "training_enabled_only",
        "chain_ids",
        "ordered_muscle_schema_sha256",
        "edge_set_fingerprint",
        "scope",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("graph regularization manifest fields differ from contract")
    if value["schema_version"] != GRAPH_REGULARIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported graph regularization schema")
    if value["method"] != LAPLACIAN_NMF_METHOD:
        raise ValueError("unsupported graph regularization method")
    if type(value["enabled"]) is not bool or value["training_enabled_only"] is not True:
        raise ValueError("graph regularization boolean contract is invalid")
    requested = _positive_finite(value["requested_lambda"], "graph requested_lambda")
    coefficient = _nonnegative_finite(value["lambda"], "graph lambda")
    edge_count = _nonnegative_int(value["edge_count"], "graph edge_count")
    full_edge_count = _nonnegative_int(value["full_graph_edge_count"], "graph full_edge_count")
    if edge_count > full_edge_count:
        raise ValueError("graph regularization subset edge_count exceeds the full graph")
    if value["enabled"] != (edge_count > 0):
        raise ValueError("graph regularization enabled flag differs from edge coverage")
    expected_lambda = requested if value["enabled"] else 0.0
    if coefficient != expected_lambda:
        raise ValueError("graph regularization lambda differs from enabled edge coverage")
    chain_ids = value["chain_ids"]
    if not isinstance(chain_ids, list) or not chain_ids:
        raise ValueError("graph regularization requires non-empty verified chain_ids")
    if any(not isinstance(item, str) or not item.strip() for item in chain_ids):
        raise ValueError("graph regularization chain_ids must be non-empty strings")
    if len(set(chain_ids)) != len(chain_ids):
        raise ValueError("graph regularization chain_ids must be unique")
    return {
        "schema_version": GRAPH_REGULARIZATION_SCHEMA_VERSION,
        "enabled": value["enabled"],
        "method": LAPLACIAN_NMF_METHOD,
        "lambda": coefficient,
        "requested_lambda": requested,
        "continuity_graph_id": _nonempty_text(value["continuity_graph_id"], "continuity_graph_id"),
        "continuity_graph_fingerprint": _sha256(
            value["continuity_graph_fingerprint"],
            "continuity_graph_fingerprint",
        ),
        "taxonomy_id": _nonempty_text(value["taxonomy_id"], "taxonomy_id"),
        "taxonomy_fingerprint": _sha256(value["taxonomy_fingerprint"], "taxonomy_fingerprint"),
        "edge_count": edge_count,
        "full_graph_edge_count": full_edge_count,
        "normalization_space": (
            GRAPH_NORMALIZATION_SPACE
            if value["normalization_space"] == GRAPH_NORMALIZATION_SPACE
            else _raise_normalization_space()
        ),
        "training_enabled_only": True,
        "chain_ids": [str(item) for item in chain_ids],
        "ordered_muscle_schema_sha256": _sha256(
            value["ordered_muscle_schema_sha256"],
            "ordered_muscle_schema_sha256",
        ),
        "edge_set_fingerprint": _sha256(value["edge_set_fingerprint"], "edge_set_fingerprint"),
        "scope": _nonempty_text(value["scope"], "scope"),
    }


def graph_regularization_lineage_fingerprint(value: Mapping[str, Any]) -> str:
    return _json_sha256(validate_graph_regularization_manifest(value))


def _validate_adjacency(value: Any, *, width: int) -> np.ndarray:
    adjacency = np.asarray(value, dtype=np.float64)
    if adjacency.shape != (width, width):
        raise ValueError(f"graph adjacency must have shape ({width}, {width})")
    if not np.all(np.isfinite(adjacency)) or np.any(adjacency < 0.0):
        raise ValueError("graph adjacency must be finite and non-negative")
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("graph adjacency must be exactly symmetric")
    if np.any(np.diag(adjacency) != 0.0):
        raise ValueError("graph adjacency diagonal must be zero")
    return np.ascontiguousarray(adjacency)


def _adjacency_edge_count(adjacency: np.ndarray) -> int:
    return int(np.count_nonzero(np.triu(adjacency, k=1)))


def _edge_set_fingerprint(adjacency: np.ndarray, names: Sequence[str]) -> str:
    edges = [
        [str(names[i]), str(names[j]), float(adjacency[i, j])]
        for i in range(adjacency.shape[0])
        for j in range(i + 1, adjacency.shape[1])
        if adjacency[i, j] > 0.0
    ]
    return _json_sha256({"schema_version": "synergy_graph_edge_set_v1", "edges": edges})


def _positive_finite(value: Any, field: str) -> float:
    result = _nonnegative_finite(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _nonnegative_finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and non-negative")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or int(value) < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, field: str) -> str:
    text = _nonempty_text(value, field)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{field} must be lowercase SHA-256")
    return text


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _raise_normalization_space() -> str:
    raise ValueError("graph regularization requires source_signal_units_no_channel_normalization")
