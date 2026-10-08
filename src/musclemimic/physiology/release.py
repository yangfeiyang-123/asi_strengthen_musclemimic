"""Immutable, single-file continuity training contracts.

A release does not embed large source artifacts.  It binds their canonical
fingerprints and paths, then the runtime re-loads and re-validates every one
before compiling a reward.  This prevents a graph, calibration coefficient,
or loss specification from being spliced across experiments.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.physiology.anatomical_groups import (
    AnatomicalTaxonomy,
    load_anatomical_taxonomy,
)
from musclemimic.physiology.continuity_groups import (
    ContinuityLossSpecIdentity,
    FascicleContinuityGraph,
    assert_continuity_loss_spec_matches,
    build_continuity_loss_spec,
    load_continuity_loss_spec_identity,
    load_fascicle_continuity_graph,
    validate_candidate_continuity_graph,
)

CONTINUITY_TRAINING_RELEASE_SCHEMA_VERSION = "continuity_training_release_v1"
ALLOWED_CONTINUITY_ACTION_MODES = (
    "full_354",
    "fixed_synergy",
    "fixed_synergy_residual",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ContinuityTrainingRelease:
    schema_version: str
    release_id: str
    taxonomy: dict[str, Any]
    diagnostic_graph: dict[str, Any]
    topology_review: dict[str, Any]
    candidate_graph: dict[str, Any]
    loss_spec: dict[str, Any]
    baseline: dict[str, Any]
    calibration: dict[str, Any]
    reward: dict[str, Any]
    allowed_action_modes: tuple[str, ...]
    created_at_utc: str
    release_fingerprint: str
    source_path: Path | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "taxonomy": copy.deepcopy(self.taxonomy),
            "diagnostic_graph": copy.deepcopy(self.diagnostic_graph),
            "topology_review": copy.deepcopy(self.topology_review),
            "candidate_graph": copy.deepcopy(self.candidate_graph),
            "loss_spec": copy.deepcopy(self.loss_spec),
            "baseline": copy.deepcopy(self.baseline),
            "calibration": copy.deepcopy(self.calibration),
            "reward": copy.deepcopy(self.reward),
            "allowed_action_modes": list(self.allowed_action_modes),
            "created_at_utc": self.created_at_utc,
            "release_fingerprint": self.release_fingerprint,
        }


@dataclass(frozen=True)
class ContinuityTrainingReleaseArtifacts:
    release: ContinuityTrainingRelease
    taxonomy: AnatomicalTaxonomy
    diagnostic_graph: FascicleContinuityGraph
    topology_review: dict[str, Any]
    candidate_graph: FascicleContinuityGraph
    loss_identity: ContinuityLossSpecIdentity
    baseline_rollout: dict[str, Any]
    rollout_manifest: dict[str, Any]
    calibration: dict[str, Any]


def continuity_training_release_fingerprint(payload: Mapping[str, Any]) -> str:
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("release_fingerprint", None)
    return _json_sha256(unsigned)


def load_continuity_training_release(path: str | Path) -> ContinuityTrainingRelease:
    source = Path(path).expanduser().resolve(strict=True)
    return validate_continuity_training_release(
        _load_json(source),
        source_path=source,
    )


def validate_continuity_training_release(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> ContinuityTrainingRelease:
    _exact_keys(
        payload,
        {
            "schema_version",
            "release_id",
            "taxonomy",
            "diagnostic_graph",
            "topology_review",
            "candidate_graph",
            "loss_spec",
            "baseline",
            "calibration",
            "reward",
            "allowed_action_modes",
            "created_at_utc",
            "release_fingerprint",
        },
        "continuity training release",
    )
    if payload["schema_version"] != CONTINUITY_TRAINING_RELEASE_SCHEMA_VERSION:
        raise ValueError("unsupported continuity training release schema")
    release_id = _text(payload["release_id"], "release_id")
    taxonomy = _validate_taxonomy_contract(payload["taxonomy"])
    diagnostic_graph = _validate_diagnostic_graph_contract(payload["diagnostic_graph"])
    review = _validate_review_contract(payload["topology_review"])
    graph = _validate_graph_contract(payload["candidate_graph"])
    loss = _validate_loss_contract(payload["loss_spec"])
    baseline = _validate_baseline_contract(payload["baseline"])
    calibration = _validate_calibration_contract(payload["calibration"])
    reward = _validate_reward_contract(payload["reward"])
    modes_raw = payload["allowed_action_modes"]
    if not isinstance(modes_raw, list) or not modes_raw:
        raise ValueError("allowed_action_modes must be a non-empty list")
    modes = tuple(_text(value, "allowed action mode") for value in modes_raw)
    if len(set(modes)) != len(modes) or any(mode not in ALLOWED_CONTINUITY_ACTION_MODES for mode in modes):
        raise ValueError("continuity release allowed_action_modes are invalid")
    if tuple(mode for mode in ALLOWED_CONTINUITY_ACTION_MODES if mode in modes) != modes:
        raise ValueError("continuity release allowed_action_modes must use canonical order")
    created_at = _utc_timestamp(payload["created_at_utc"])

    if review["review_fingerprint"] != graph["topology_review_fingerprint"]:
        raise ValueError("release topology review differs from candidate graph lineage")
    if diagnostic_graph["graph_fingerprint"] != review["source_graph_fingerprint"]:
        raise ValueError("release diagnostic graph differs from topology review")
    if graph["graph_fingerprint"] != loss["graph_fingerprint"]:
        raise ValueError("release candidate graph differs from loss spec")
    if loss["loss_spec_fingerprint"] != calibration["candidate_loss_spec_fingerprint"]:
        raise ValueError("release loss spec differs from calibration")
    if graph["graph_fingerprint"] != calibration["candidate_graph_fingerprint"]:
        raise ValueError("release candidate graph differs from calibration")
    if loss["target_chain_count"] != calibration["target_chain_count"]:
        raise ValueError("release target chain coverage differs from calibration")
    if loss["target_edge_count"] != calibration["target_edge_count"]:
        raise ValueError("release target edge coverage differs from calibration")
    if calibration["selected_reward_coefficient"] != reward["coefficient"]:
        raise ValueError("release reward coefficient differs from calibration")

    supplied = _sha256(payload["release_fingerprint"], "release_fingerprint")
    result = ContinuityTrainingRelease(
        schema_version=CONTINUITY_TRAINING_RELEASE_SCHEMA_VERSION,
        release_id=release_id,
        taxonomy=taxonomy,
        diagnostic_graph=diagnostic_graph,
        topology_review=review,
        candidate_graph=graph,
        loss_spec=loss,
        baseline=baseline,
        calibration=calibration,
        reward=reward,
        allowed_action_modes=modes,
        created_at_utc=created_at,
        release_fingerprint=supplied,
        source_path=source_path,
    )
    if continuity_training_release_fingerprint(result.to_manifest()) != supplied:
        raise ValueError("continuity training release fingerprint is stale")
    return result


def resolve_continuity_training_release(
    release: ContinuityTrainingRelease,
) -> ContinuityTrainingReleaseArtifacts:
    """Load and cross-check every artifact referenced by ``release``."""

    from analysis.physiology_synergy.calibrate_continuity_reward import (
        validate_baseline_rollout_against_manifest,
        validate_continuity_reward_calibration,
    )
    from analysis.physiology_synergy.collect_continuity_baseline import (
        validate_rollout_manifest,
    )
    from analysis.physiology_synergy.review_continuity_topology import (
        validate_topology_review,
    )

    taxonomy = load_anatomical_taxonomy(_artifact_path(release, release.taxonomy["path_hint"]))
    diagnostic_graph = load_fascicle_continuity_graph(
        _artifact_path(release, release.diagnostic_graph["artifact_path"]),
        taxonomy=taxonomy,
    )
    if diagnostic_graph.training_enabled_chain_count:
        raise ValueError("released diagnostic graph must remain diagnostics-only")
    review = validate_topology_review(
        _load_json(_artifact_path(release, release.topology_review["artifact_path"])),
        source_graph=diagnostic_graph,
        taxonomy=taxonomy,
    )
    graph = load_fascicle_continuity_graph(
        _artifact_path(release, release.candidate_graph["artifact_path"]),
        taxonomy=taxonomy,
    )
    validate_candidate_continuity_graph(
        graph,
        taxonomy,
        expected_review_fingerprint=review["review_fingerprint"],
        source_graph=diagnostic_graph,
    )
    loss_identity = load_continuity_loss_spec_identity(_artifact_path(release, release.loss_spec["artifact_path"]))
    baseline = _load_json(_artifact_path(release, release.baseline["rollout_artifact_path"]))
    rollout_manifest = validate_rollout_manifest(
        _load_json(_artifact_path(release, release.baseline["rollout_manifest_path"]))
    )
    baseline = validate_baseline_rollout_against_manifest(baseline, rollout_manifest)
    calibration = validate_continuity_reward_calibration(
        _load_json(_artifact_path(release, release.calibration["artifact_path"]))
    )

    stable = taxonomy.stable_model_binding
    expected_taxonomy = {
        "taxonomy_id": taxonomy.taxonomy_id,
        "taxonomy_fingerprint": taxonomy.fingerprint,
        "actuator_schema_hash": stable["actuator_schema_hash"],
        "muscle_channel_core_fingerprint": stable["muscle_channel_core_fingerprint"],
    }
    for field, expected in expected_taxonomy.items():
        if release.taxonomy[field] != expected:
            raise ValueError(f"release taxonomy {field} differs from artifact")
    if release.topology_review["review_fingerprint"] != review["review_fingerprint"]:
        raise ValueError("release topology review fingerprint differs from artifact")
    expected_diagnostic = {
        "graph_id": diagnostic_graph.graph_id,
        "graph_fingerprint": diagnostic_graph.graph_fingerprint,
        "global_chain_count": len(diagnostic_graph.chains),
        "global_edge_count": diagnostic_graph.edge_count,
    }
    for field, expected in expected_diagnostic.items():
        if release.diagnostic_graph[field] != expected:
            raise ValueError(f"release diagnostic graph {field} differs from artifact")
    candidate_lineage = graph.generation["candidate_graph"]
    if review["source_graph_fingerprint"] != candidate_lineage["source_graph_fingerprint"]:
        raise ValueError("topology review source graph differs from candidate lineage")
    _validate_reviewed_candidate_parameters(review, graph)

    expected_graph = {
        "graph_id": graph.graph_id,
        "graph_fingerprint": graph.graph_fingerprint,
        "topology_review_fingerprint": candidate_lineage["topology_review_fingerprint"],
        "training_chain_ids": list(loss_identity.chain_ids),
    }
    for field, expected in expected_graph.items():
        if release.candidate_graph[field] != expected:
            raise ValueError(f"release candidate graph {field} differs from artifact")

    _, runtime_identity = build_continuity_loss_spec(
        graph,
        taxonomy,
        training_enabled_only=True,
        signal=loss_identity.signal,
        method=loss_identity.method,
        scale=loss_identity.scale,
        huber_delta=loss_identity.huber_delta,
        eps=loss_identity.eps,
    )
    assert_continuity_loss_spec_matches(loss_identity, runtime_identity)
    _validate_release_loss_fields(release, loss_identity)

    if release.baseline["rollout_artifact_fingerprint"] != baseline["artifact_fingerprint"]:
        raise ValueError("release baseline fingerprint differs from artifact")
    if release.baseline["rollout_manifest_fingerprint"] != rollout_manifest["rollout_manifest_fingerprint"]:
        raise ValueError("release rollout manifest fingerprint differs from artifact")
    if calibration["source_baseline_rollout_fingerprint"] != baseline["artifact_fingerprint"]:
        raise ValueError("calibration does not consume the released baseline")
    if release.calibration["calibration_fingerprint"] != calibration["calibration_fingerprint"]:
        raise ValueError("release calibration fingerprint differs from artifact")
    expected_calibration = {
        "candidate_graph_fingerprint": calibration["identity"]["candidate_graph_fingerprint"],
        "candidate_loss_spec_fingerprint": calibration["identity"]["candidate_loss_spec_fingerprint"],
        "target_chain_count": calibration["coverage"]["target_chain_count"],
        "target_edge_count": calibration["coverage"]["target_edge_count"],
        "selected_budget_fraction": calibration["selection"]["fraction_of_median_imitation_reward"],
        "selected_reward_coefficient": calibration["selection"]["coefficient"],
    }
    for field, expected in expected_calibration.items():
        if release.calibration[field] != expected:
            raise ValueError(f"release calibration {field} differs from artifact")
    if baseline["identity"]["candidate_loss_spec_fingerprint"] != loss_identity.loss_spec_fingerprint:
        raise ValueError("released baseline target loss differs from loss spec")
    if baseline["identity"]["diagnostic_graph_fingerprint"] != diagnostic_graph.graph_fingerprint:
        raise ValueError("released baseline diagnostic graph differs from release")

    return ContinuityTrainingReleaseArtifacts(
        release=release,
        taxonomy=taxonomy,
        diagnostic_graph=diagnostic_graph,
        topology_review=review,
        candidate_graph=graph,
        loss_identity=loss_identity,
        baseline_rollout=baseline,
        rollout_manifest=rollout_manifest,
        calibration=calibration,
    )


def validate_release_against_runtime(
    release: ContinuityTrainingRelease,
    *,
    taxonomy: AnatomicalTaxonomy,
    graph: FascicleContinuityGraph,
    runtime_loss_identity: ContinuityLossSpecIdentity,
    action_mode: str,
) -> None:
    if action_mode not in release.allowed_action_modes:
        raise ValueError(f"continuity release does not allow action mode {action_mode!r}")
    if release.taxonomy["taxonomy_fingerprint"] != taxonomy.fingerprint:
        raise ValueError("continuity release taxonomy differs from runtime")
    if release.taxonomy["actuator_schema_hash"] != taxonomy.stable_model_binding["actuator_schema_hash"]:
        raise ValueError("continuity release actuator schema differs from runtime")
    if (
        release.taxonomy["muscle_channel_core_fingerprint"]
        != taxonomy.stable_model_binding["muscle_channel_core_fingerprint"]
    ):
        raise ValueError("continuity release muscle channel ABI differs from runtime")
    if release.candidate_graph["graph_fingerprint"] != graph.graph_fingerprint:
        raise ValueError("continuity release candidate graph differs from runtime")
    if release.loss_spec["loss_spec_fingerprint"] != runtime_loss_identity.loss_spec_fingerprint:
        raise ValueError("continuity release loss spec differs from runtime")
    if release.loss_spec["target_chain_count"] != runtime_loss_identity.chain_count:
        raise ValueError("continuity release runtime chain coverage differs")
    if release.loss_spec["target_edge_count"] != runtime_loss_identity.edge_count:
        raise ValueError("continuity release runtime edge coverage differs")


def _validate_release_loss_fields(
    release: ContinuityTrainingRelease,
    identity: ContinuityLossSpecIdentity,
) -> None:
    expected = {
        "loss_spec_fingerprint": identity.loss_spec_fingerprint,
        "graph_fingerprint": identity.graph_fingerprint,
        "signal": identity.signal,
        "method": identity.method,
        "scale": identity.scale,
        "huber_delta": identity.huber_delta,
        "eps": identity.eps,
        "reduction": identity.reduction,
        "normalization": identity.normalization,
        "target_chain_count": identity.chain_count,
        "target_edge_count": identity.edge_count,
    }
    for field, expected_value in expected.items():
        if release.loss_spec[field] != expected_value:
            raise ValueError(f"release loss spec {field} differs from artifact")


def _validate_reviewed_candidate_parameters(
    review: Mapping[str, Any],
    graph: FascicleContinuityGraph,
) -> None:
    graph_by_id = {chain["chain_id"]: chain for chain in graph.chains}
    approved = [entry for entry in review["chains"] if entry["approve_as_training_candidate"]]
    if [entry["chain_id"] for entry in approved] != [
        chain["chain_id"] for chain in graph.chains if chain["training_enabled"]
    ]:
        raise ValueError("candidate training chains differ from topology review")
    field_pairs = (
        ("approved_deadband", "deadband"),
        ("approved_edge_weights", "edge_weights"),
        ("approved_chain_weight", "chain_weight"),
        ("approved_activity_off", "activity_off"),
        ("approved_activity_on", "activity_on"),
    )
    for entry in approved:
        chain = graph_by_id[entry["chain_id"]]
        for review_field, graph_field in field_pairs:
            if entry[review_field] != chain[graph_field]:
                raise ValueError(f"candidate chain {entry['chain_id']} differs from reviewed {graph_field}")


def _validate_taxonomy_contract(value: Any) -> dict[str, Any]:
    result = _mapping(value, "release taxonomy")
    _exact_keys(
        result,
        {
            "path_hint",
            "taxonomy_id",
            "taxonomy_fingerprint",
            "actuator_schema_hash",
            "muscle_channel_core_fingerprint",
        },
        "release taxonomy",
    )
    result["path_hint"] = _text(result["path_hint"], "taxonomy.path_hint")
    result["taxonomy_id"] = _text(result["taxonomy_id"], "taxonomy.taxonomy_id")
    for field in ("taxonomy_fingerprint", "actuator_schema_hash", "muscle_channel_core_fingerprint"):
        result[field] = _sha256(result[field], f"taxonomy.{field}")
    return result


def _validate_review_contract(value: Any) -> dict[str, Any]:
    result = _mapping(value, "release topology review")
    _exact_keys(
        result,
        {"artifact_path", "source_graph_fingerprint", "review_fingerprint"},
        "release topology review",
    )
    result["artifact_path"] = _text(result["artifact_path"], "topology_review.artifact_path")
    result["source_graph_fingerprint"] = _sha256(
        result["source_graph_fingerprint"],
        "topology_review.source_graph_fingerprint",
    )
    result["review_fingerprint"] = _sha256(result["review_fingerprint"], "review_fingerprint")
    return result


def _validate_diagnostic_graph_contract(value: Any) -> dict[str, Any]:
    result = _mapping(value, "release diagnostic graph")
    _exact_keys(
        result,
        {
            "artifact_path",
            "graph_id",
            "graph_fingerprint",
            "global_chain_count",
            "global_edge_count",
        },
        "release diagnostic graph",
    )
    for field in ("artifact_path", "graph_id"):
        result[field] = _text(result[field], f"diagnostic_graph.{field}")
    result["graph_fingerprint"] = _sha256(
        result["graph_fingerprint"],
        "diagnostic_graph.graph_fingerprint",
    )
    for field in ("global_chain_count", "global_edge_count"):
        result[field] = _positive_int(result[field], f"diagnostic_graph.{field}")
    return result


def _validate_graph_contract(value: Any) -> dict[str, Any]:
    result = _mapping(value, "release candidate graph")
    _exact_keys(
        result,
        {
            "artifact_path",
            "graph_id",
            "graph_fingerprint",
            "topology_review_fingerprint",
            "training_chain_ids",
        },
        "release candidate graph",
    )
    for field in ("artifact_path", "graph_id"):
        result[field] = _text(result[field], f"candidate_graph.{field}")
    for field in ("graph_fingerprint", "topology_review_fingerprint"):
        result[field] = _sha256(result[field], f"candidate_graph.{field}")
    ids = result["training_chain_ids"]
    if not isinstance(ids, list) or not ids:
        raise ValueError("candidate_graph.training_chain_ids must be non-empty")
    result["training_chain_ids"] = [_text(value, "training chain id") for value in ids]
    if len(set(result["training_chain_ids"])) != len(result["training_chain_ids"]):
        raise ValueError("candidate_graph.training_chain_ids contain duplicates")
    return result


def _validate_loss_contract(value: Any) -> dict[str, Any]:
    result = _mapping(value, "release loss spec")
    _exact_keys(
        result,
        {
            "artifact_path",
            "loss_spec_fingerprint",
            "graph_fingerprint",
            "signal",
            "method",
            "scale",
            "huber_delta",
            "eps",
            "reduction",
            "normalization",
            "target_chain_count",
            "target_edge_count",
        },
        "release loss spec",
    )
    result["artifact_path"] = _text(result["artifact_path"], "loss_spec.artifact_path")
    for field in ("loss_spec_fingerprint", "graph_fingerprint"):
        result[field] = _sha256(result[field], f"loss_spec.{field}")
    for field in ("signal", "method", "reduction", "normalization"):
        result[field] = _text(result[field], f"loss_spec.{field}")
    for field in ("scale", "huber_delta", "eps"):
        result[field] = _positive_finite(result[field], f"loss_spec.{field}")
    for field in ("target_chain_count", "target_edge_count"):
        result[field] = _positive_int(result[field], f"loss_spec.{field}")
    return result


def _validate_baseline_contract(value: Any) -> dict[str, Any]:
    result = _mapping(value, "release baseline")
    _exact_keys(
        result,
        {
            "rollout_artifact_path",
            "rollout_artifact_fingerprint",
            "rollout_manifest_path",
            "rollout_manifest_fingerprint",
        },
        "release baseline",
    )
    for field in ("rollout_artifact_path", "rollout_manifest_path"):
        result[field] = _text(result[field], f"baseline.{field}")
    for field in ("rollout_artifact_fingerprint", "rollout_manifest_fingerprint"):
        result[field] = _sha256(result[field], f"baseline.{field}")
    return result


def _validate_calibration_contract(value: Any) -> dict[str, Any]:
    result = _mapping(value, "release calibration")
    _exact_keys(
        result,
        {
            "artifact_path",
            "calibration_fingerprint",
            "candidate_graph_fingerprint",
            "candidate_loss_spec_fingerprint",
            "target_chain_count",
            "target_edge_count",
            "selected_budget_fraction",
            "selected_reward_coefficient",
        },
        "release calibration",
    )
    result["artifact_path"] = _text(result["artifact_path"], "calibration.artifact_path")
    for field in (
        "calibration_fingerprint",
        "candidate_graph_fingerprint",
        "candidate_loss_spec_fingerprint",
    ):
        result[field] = _sha256(result[field], f"calibration.{field}")
    for field in ("target_chain_count", "target_edge_count"):
        result[field] = _positive_int(result[field], f"calibration.{field}")
    for field in ("selected_budget_fraction", "selected_reward_coefficient"):
        result[field] = _positive_finite(result[field], f"calibration.{field}")
    return result


def _validate_reward_contract(value: Any) -> dict[str, Any]:
    result = _mapping(value, "release reward")
    _exact_keys(result, {"coefficient", "raw_penalty_clip"}, "release reward")
    result["coefficient"] = _positive_finite(result["coefficient"], "reward.coefficient")
    clip = result["raw_penalty_clip"]
    result["raw_penalty_clip"] = None if clip is None else _positive_finite(clip, "reward.raw_penalty_clip")
    return result


def _artifact_path(release: ContinuityTrainingRelease, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        if release.source_path is None:
            raise ValueError("relative release artifact path requires a release source path")
        path = release.source_path.parent / path
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"continuity release artifact is missing: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON value in {path}: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(payload)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return copy.deepcopy(dict(value))


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{field} fields differ from contract")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, field: str) -> str:
    result = _text(value, field)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{field} must be lowercase SHA-256")
    return result


def _positive_finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and positive")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0 or result != float(value):
        raise ValueError(f"{field} must be a positive integer")
    return result


def _utc_timestamp(value: Any) -> str:
    from datetime import datetime

    text = _text(value, "created_at_utc")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at_utc must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0.0:
        raise ValueError("created_at_utc must use an explicit UTC offset")
    return parsed.isoformat().replace("+00:00", "Z")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
