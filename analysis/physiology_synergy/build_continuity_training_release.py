"""Finalize reviewed continuity artifacts into one immutable training release."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from musclemimic.physiology.anatomical_groups import load_anatomical_taxonomy
from musclemimic.physiology.continuity_groups import (
    assert_continuity_loss_spec_matches,
    build_continuity_loss_spec,
    load_continuity_loss_spec_identity,
    load_fascicle_continuity_graph,
    validate_candidate_continuity_graph,
)
from musclemimic.physiology.release import (
    ALLOWED_CONTINUITY_ACTION_MODES,
    CONTINUITY_TRAINING_RELEASE_SCHEMA_VERSION,
    continuity_training_release_fingerprint,
    resolve_continuity_training_release,
    validate_continuity_training_release,
)


def build_continuity_training_release(
    *,
    taxonomy_path: str | Path,
    diagnostic_graph_path: str | Path | None = None,
    topology_review_path: str | Path,
    candidate_graph_path: str | Path,
    loss_spec_path: str | Path,
    baseline_rollout_path: str | Path,
    rollout_manifest_path: str | Path,
    calibration_path: str | Path,
    release_id: str | None = None,
    allowed_action_modes: Sequence[str] = ALLOWED_CONTINUITY_ACTION_MODES,
    raw_penalty_clip: float | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    if diagnostic_graph_path is None:
        diagnostic_graph_path = (
            Path(__file__).resolve().parents[2] / "configs/physiology/myofullbody_354_fascicle_continuity_v2.json"
        )
    paths = {
        "taxonomy": _resolve(taxonomy_path),
        "diagnostic": _resolve(diagnostic_graph_path),
        "review": _resolve(topology_review_path),
        "graph": _resolve(candidate_graph_path),
        "loss": _resolve(loss_spec_path),
        "baseline": _resolve(baseline_rollout_path),
        "manifest": _resolve(rollout_manifest_path),
        "calibration": _resolve(calibration_path),
    }
    taxonomy = load_anatomical_taxonomy(paths["taxonomy"])
    if not taxonomy.release_eligible:
        raise ValueError("continuity training releases require anatomical taxonomy v2")
    diagnostic_graph = load_fascicle_continuity_graph(
        paths["diagnostic"],
        taxonomy=taxonomy,
    )
    if diagnostic_graph.training_enabled_chain_count:
        raise ValueError("continuity release diagnostic graph must remain diagnostics-only")
    review = validate_topology_review(
        _load_json(paths["review"]),
        source_graph=diagnostic_graph,
        taxonomy=taxonomy,
    )
    graph = load_fascicle_continuity_graph(paths["graph"], taxonomy=taxonomy)
    validate_candidate_continuity_graph(
        graph,
        taxonomy,
        expected_review_fingerprint=review["review_fingerprint"],
        source_graph=diagnostic_graph,
    )
    loss_identity = load_continuity_loss_spec_identity(paths["loss"])
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
    rollout_manifest = validate_rollout_manifest(_load_json(paths["manifest"]))
    baseline = validate_baseline_rollout_against_manifest(
        _load_json(paths["baseline"]),
        rollout_manifest,
    )
    calibration = validate_continuity_reward_calibration(_load_json(paths["calibration"]))
    if calibration["source_baseline_rollout_fingerprint"] != baseline["artifact_fingerprint"]:
        raise ValueError("calibration does not consume the supplied baseline")
    if calibration["identity"]["candidate_graph_fingerprint"] != graph.graph_fingerprint:
        raise ValueError("calibration candidate graph differs from supplied graph")
    if calibration["identity"]["candidate_loss_spec_fingerprint"] != loss_identity.loss_spec_fingerprint:
        raise ValueError("calibration loss spec differs from supplied loss spec")

    stable = taxonomy.stable_model_binding
    candidate_lineage = graph.generation["candidate_graph"]
    timestamp = created_at_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    resolved_release_id = release_id or (f"{graph.graph_id}_cal_{calibration['calibration_fingerprint'][:12]}")
    selected = calibration["selection"]
    payload: dict[str, Any] = {
        "schema_version": CONTINUITY_TRAINING_RELEASE_SCHEMA_VERSION,
        "release_id": resolved_release_id,
        "taxonomy": {
            "path_hint": str(paths["taxonomy"]),
            "taxonomy_id": taxonomy.taxonomy_id,
            "taxonomy_fingerprint": taxonomy.fingerprint,
            "actuator_schema_hash": stable["actuator_schema_hash"],
            "muscle_channel_core_fingerprint": stable["muscle_channel_core_fingerprint"],
        },
        "diagnostic_graph": {
            "artifact_path": str(paths["diagnostic"]),
            "graph_id": diagnostic_graph.graph_id,
            "graph_fingerprint": diagnostic_graph.graph_fingerprint,
            "global_chain_count": len(diagnostic_graph.chains),
            "global_edge_count": diagnostic_graph.edge_count,
        },
        "topology_review": {
            "artifact_path": str(paths["review"]),
            "source_graph_fingerprint": review["source_graph_fingerprint"],
            "review_fingerprint": review["review_fingerprint"],
        },
        "candidate_graph": {
            "artifact_path": str(paths["graph"]),
            "graph_id": graph.graph_id,
            "graph_fingerprint": graph.graph_fingerprint,
            "topology_review_fingerprint": candidate_lineage["topology_review_fingerprint"],
            "training_chain_ids": list(loss_identity.chain_ids),
        },
        "loss_spec": {
            "artifact_path": str(paths["loss"]),
            "loss_spec_fingerprint": loss_identity.loss_spec_fingerprint,
            "graph_fingerprint": loss_identity.graph_fingerprint,
            "signal": loss_identity.signal,
            "method": loss_identity.method,
            "scale": loss_identity.scale,
            "huber_delta": loss_identity.huber_delta,
            "eps": loss_identity.eps,
            "reduction": loss_identity.reduction,
            "normalization": loss_identity.normalization,
            "target_chain_count": loss_identity.chain_count,
            "target_edge_count": loss_identity.edge_count,
        },
        "baseline": {
            "rollout_artifact_path": str(paths["baseline"]),
            "rollout_artifact_fingerprint": baseline["artifact_fingerprint"],
            "rollout_manifest_path": str(paths["manifest"]),
            "rollout_manifest_fingerprint": rollout_manifest["rollout_manifest_fingerprint"],
        },
        "calibration": {
            "artifact_path": str(paths["calibration"]),
            "calibration_fingerprint": calibration["calibration_fingerprint"],
            "candidate_graph_fingerprint": calibration["identity"]["candidate_graph_fingerprint"],
            "candidate_loss_spec_fingerprint": calibration["identity"]["candidate_loss_spec_fingerprint"],
            "target_chain_count": calibration["coverage"]["target_chain_count"],
            "target_edge_count": calibration["coverage"]["target_edge_count"],
            "selected_budget_fraction": selected["fraction_of_median_imitation_reward"],
            "selected_reward_coefficient": selected["coefficient"],
        },
        "reward": {
            "coefficient": selected["coefficient"],
            "raw_penalty_clip": raw_penalty_clip,
        },
        "allowed_action_modes": list(allowed_action_modes),
        "created_at_utc": timestamp,
    }
    payload["release_fingerprint"] = continuity_training_release_fingerprint(payload)
    release = validate_continuity_training_release(payload)
    resolve_continuity_training_release(release)
    return release.to_manifest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy-json", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-graph-json",
        type=Path,
        default=None,
        help="Full diagnostics graph; defaults to the checked-in MyoFullBody v2 graph.",
    )
    parser.add_argument("--topology-review-json", type=Path, required=True)
    parser.add_argument("--candidate-graph-json", type=Path, required=True)
    parser.add_argument("--loss-spec-json", type=Path, required=True)
    parser.add_argument("--baseline-rollout-json", type=Path, required=True)
    parser.add_argument("--rollout-manifest-json", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--release-id", default=None)
    parser.add_argument(
        "--allowed-action-mode",
        action="append",
        choices=ALLOWED_CONTINUITY_ACTION_MODES,
        dest="allowed_action_modes",
    )
    parser.add_argument("--raw-penalty-clip", type=float, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite continuity release: {args.output_json}")
    payload = build_continuity_training_release(
        taxonomy_path=args.taxonomy_json,
        diagnostic_graph_path=args.diagnostic_graph_json,
        topology_review_path=args.topology_review_json,
        candidate_graph_path=args.candidate_graph_json,
        loss_spec_path=args.loss_spec_json,
        baseline_rollout_path=args.baseline_rollout_json,
        rollout_manifest_path=args.rollout_manifest_json,
        calibration_path=args.calibration_json,
        release_id=args.release_id,
        allowed_action_modes=(
            ALLOWED_CONTINUITY_ACTION_MODES if args.allowed_action_modes is None else tuple(args.allowed_action_modes)
        ),
        raw_penalty_clip=args.raw_penalty_clip,
    )
    _atomic_write_json(args.output_json, payload)
    print(args.output_json.resolve())


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=True)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(payload)


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
