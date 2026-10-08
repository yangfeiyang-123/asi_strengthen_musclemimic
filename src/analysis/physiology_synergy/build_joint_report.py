"""Combine task, tracking, continuity, control-synergy and EMG evidence.

The report is an evidence index, not a result generator.  Every section keeps
the source artifact fingerprint, and identities shared by two inputs must agree
before a report can be emitted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from musclemimic.runner.checkpointing import config_hash as experiment_config_hash

JOINT_REPORT_SCHEMA_VERSION = "forehand_physio_synergy_joint_report_v2"
ROLLOUT_METRICS_EVIDENCE_SCHEMA_VERSION = "source_bound_rollout_metrics_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


def joint_report_fingerprint(payload: Mapping[str, Any]) -> str:
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("report_fingerprint", None)
    return _canonical_json_sha256(unsigned)


def rollout_metrics_evidence_fingerprint(payload: Mapping[str, Any]) -> str:
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("artifact_fingerprint", None)
    return _canonical_json_sha256(unsigned)


def build_rollout_metrics_evidence(
    *,
    identity: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": ROLLOUT_METRICS_EVIDENCE_SCHEMA_VERSION,
        "identity": copy.deepcopy(dict(identity)),
        "metrics": copy.deepcopy(dict(metrics)),
    }
    payload["artifact_fingerprint"] = rollout_metrics_evidence_fingerprint(payload)
    return validate_rollout_metrics_evidence(payload)


def validate_rollout_metrics_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(value, "rollout_metrics")
    expected = {"schema_version", "identity", "metrics", "artifact_fingerprint"}
    if set(payload) != expected:
        raise ValueError("rollout metrics evidence fields differ from contract")
    if payload["schema_version"] != ROLLOUT_METRICS_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported rollout metrics evidence schema")
    identity = _mapping(payload["identity"], "rollout_metrics.identity")
    expected_identity = {
        "run_id",
        "config_hash",
        "checkpoint_fingerprint",
        "promotion_fingerprint",
        "formal_synergy_basis_fingerprint",
        "taxonomy_fingerprint",
        "continuity_graph_fingerprint",
        "event_reference_fingerprint",
        "session_uid",
        "evaluation_split_fingerprint",
        "environment_fingerprint",
    }
    if set(identity) != expected_identity:
        raise ValueError("rollout metrics identity fields differ from contract")
    canonical_identity = {
        "run_id": _nonempty_text(identity["run_id"], "rollout metrics run_id"),
        "config_hash": _nonempty_text(identity["config_hash"], "rollout metrics config_hash"),
        **{
            field: _require_sha256(identity[field], f"rollout metrics {field}")
            for field in (
                "checkpoint_fingerprint",
                "promotion_fingerprint",
                "formal_synergy_basis_fingerprint",
                "taxonomy_fingerprint",
                "continuity_graph_fingerprint",
                "event_reference_fingerprint",
                "evaluation_split_fingerprint",
                "environment_fingerprint",
            )
        },
        "session_uid": _nonempty_text(identity["session_uid"], "rollout metrics session_uid"),
    }
    metrics = _mapping(payload["metrics"], "rollout_metrics.metrics")
    if not metrics:
        raise ValueError("rollout metrics evidence cannot be empty")
    canonical_metrics = _numeric_metric_tree(metrics, "rollout_metrics.metrics")
    result = {
        "schema_version": ROLLOUT_METRICS_EVIDENCE_SCHEMA_VERSION,
        "identity": canonical_identity,
        "metrics": canonical_metrics,
        "artifact_fingerprint": _require_sha256(
            payload["artifact_fingerprint"],
            "rollout metrics artifact_fingerprint",
        ),
    }
    if rollout_metrics_evidence_fingerprint(result) != result["artifact_fingerprint"]:
        raise ValueError("rollout metrics evidence fingerprint is stale")
    return result


def build_joint_report(
    *,
    rollout_metrics: Mapping[str, Any],
    physiology_report: Mapping[str, Any],
    synergy_basis_manifest: Mapping[str, Any],
    frozen_decoder_manifest: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    checkpoint_evidence: Mapping[str, Any],
    branch_commit_sha: str,
    emg_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one fail-closed, cross-bound report from existing evidence."""

    rollout_evidence = validate_rollout_metrics_evidence(rollout_metrics)
    sources = {
        "rollout_metrics": rollout_evidence,
        "physiology_report": _mapping(physiology_report, "physiology_report"),
        "synergy_basis_manifest": _mapping(
            synergy_basis_manifest,
            "synergy_basis_manifest",
        ),
        "frozen_decoder_manifest": _mapping(
            frozen_decoder_manifest,
            "frozen_decoder_manifest",
        ),
        "resolved_config": _mapping(resolved_config, "resolved_config"),
        "checkpoint_evidence": _mapping(
            checkpoint_evidence,
            "checkpoint_evidence",
        ),
    }
    if emg_report is not None:
        sources["emg_report"] = _mapping(emg_report, "emg_report")
    source_fingerprints = {name: _canonical_json_sha256(value) for name, value in sources.items()}

    commit = str(branch_commit_sha).strip().lower()
    if _GIT_SHA.fullmatch(commit) is None:
        raise ValueError("branch_commit_sha must be a lowercase 40- or 64-hex git id")
    evidence_commit = checkpoint_evidence.get("branch_commit_sha")
    if evidence_commit not in (None, "") and str(evidence_commit).lower() != commit:
        raise ValueError("branch commit differs from checkpoint evidence")

    declared_config_hash = _nonempty_text(
        checkpoint_evidence.get("config_hash"),
        "checkpoint_evidence.config_hash",
    )
    resolved_experiment = _mapping(
        resolved_config.get("experiment"),
        "resolved_config.experiment",
    )
    computed_config_hash = experiment_config_hash(resolved_experiment)
    if declared_config_hash != computed_config_hash:
        raise ValueError("resolved experiment config hash differs from checkpoint evidence")
    checkpoint = _sha256_from(
        checkpoint_evidence,
        ("checkpoint_fingerprint", "policy_checkpoint_fingerprint"),
        "checkpoint fingerprint",
    )
    promotion = _sha256_from(
        checkpoint_evidence,
        (
            "promotion_fingerprint",
            "promotion_metrics_fingerprint",
            "policy_promotion_fingerprint",
        ),
        "promotion fingerprint",
    )
    run_id = _nonempty_text(
        checkpoint_evidence.get("run_id"),
        "checkpoint_evidence.run_id",
    )

    lineage = _mapping(
        physiology_report.get("lineage"),
        "physiology_report.lineage",
    )
    physiology_policy = _mapping(
        lineage.get("policy_evidence"),
        "physiology_report.lineage.policy_evidence",
    )
    _same_sha256(
        checkpoint,
        physiology_policy.get("policy_checkpoint_fingerprint"),
        "physiology checkpoint",
    )
    _same_sha256(
        promotion,
        physiology_policy.get("policy_promotion_fingerprint"),
        "physiology promotion",
    )

    continuity = _mapping(
        physiology_report.get("fascicle_continuity"),
        "physiology_report.fascicle_continuity",
    )
    taxonomy_binding = _mapping(
        continuity.get("taxonomy_binding"),
        "fascicle_continuity.taxonomy_binding",
    )
    taxonomy_fingerprint = _sha256_from(
        taxonomy_binding,
        ("taxonomy_fingerprint",),
        "taxonomy fingerprint",
    )
    continuity_fingerprint = _require_sha256(
        continuity.get("graph_fingerprint"),
        "continuity graph fingerprint",
    )

    formal_basis = _sha256_from(
        synergy_basis_manifest,
        ("artifact_fingerprint", "basis_fingerprint"),
        "formal synergy basis fingerprint",
    )
    _same_sha256(
        formal_basis,
        physiology_policy.get("formal_synergy_basis_fingerprint"),
        "physiology formal synergy basis",
    )
    decoder_basis = _first_present(
        frozen_decoder_manifest,
        ("basis_fingerprint", "formal_basis_fingerprint"),
    )
    if decoder_basis is None:
        raise ValueError("frozen decoder manifest has no formal basis fingerprint")
    _same_sha256(formal_basis, decoder_basis, "decoder formal synergy basis")
    decoder_fingerprint = _declared_or_canonical_fingerprint(
        frozen_decoder_manifest,
        ("decoder_fingerprint", "action_manifest_fingerprint", "manifest_fingerprint"),
        "frozen decoder",
    )

    event_reference = _require_sha256(
        lineage.get("event_reference_fingerprint"),
        "physiology event reference fingerprint",
    )
    session_uid = _nonempty_text(
        lineage.get("session_uid"),
        "physiology session_uid",
    )
    dataset_fingerprint = _sha256_from(
        synergy_basis_manifest,
        ("source_dataset_fingerprint", "dataset_fingerprint"),
        "synergy source dataset fingerprint",
    )
    split_fingerprint = _split_fingerprint(synergy_basis_manifest)

    rollout_identity = rollout_evidence["identity"]
    expected_rollout_identity = {
        "run_id": run_id,
        "config_hash": declared_config_hash,
        "checkpoint_fingerprint": checkpoint,
        "promotion_fingerprint": promotion,
        "formal_synergy_basis_fingerprint": formal_basis,
        "taxonomy_fingerprint": taxonomy_fingerprint,
        "continuity_graph_fingerprint": continuity_fingerprint,
        "event_reference_fingerprint": event_reference,
        "session_uid": session_uid,
    }
    for field, expected in expected_rollout_identity.items():
        if rollout_identity[field] != expected:
            raise ValueError(f"rollout metrics {field} differs from the selected joint-report evidence")

    emg_identity, emg_comparison, mapped_synergy, emg_limitations = _bind_emg(
        emg_report,
        checkpoint=checkpoint,
        promotion=promotion,
        formal_basis=formal_basis,
        taxonomy_fingerprint=taxonomy_fingerprint,
    )

    continuity_coverage = _mapping(
        continuity.get("coverage"),
        "fascicle_continuity.coverage",
    )
    training_chain_count = _nonnegative_int(
        continuity_coverage.get("training_enabled_chain_count"),
        "training_enabled_chain_count",
    )
    measured_chain_count = _nonnegative_int(
        continuity_coverage.get("measured_chain_count"),
        "measured_chain_count",
    )
    measured_edge_count = _nonnegative_int(
        continuity_coverage.get("measured_edge_count"),
        "measured_edge_count",
    )
    if measured_chain_count == 0 or measured_edge_count == 0:
        raise ValueError("joint report requires non-empty measured continuity coverage")

    continuity_config = _nested_mapping(
        resolved_config,
        ("experiment", "env_params", "reward_params", "intra_muscle_consistency"),
    )
    configured_mode = str(continuity_config.get("mode", "off"))
    if configured_mode not in {"off", "diagnostics", "reward"}:
        raise ValueError("joint report continuity mode is invalid")
    configured_coefficient = _finite_number(
        continuity_config.get("coefficient", 0.0),
        "continuity coefficient",
    )
    if configured_mode in {"off", "diagnostics"} and configured_coefficient != 0.0:
        raise ValueError(f"{configured_mode} config in joint report has a non-zero coefficient")
    if configured_mode in {"diagnostics", "reward"}:
        _same_sha256(
            taxonomy_fingerprint,
            continuity_config.get("expected_taxonomy_fingerprint"),
            "resolved config taxonomy",
        )
        _same_sha256(
            continuity_fingerprint,
            continuity_config.get("expected_continuity_fingerprint"),
            "resolved config continuity graph",
        )
    if configured_mode == "reward" and training_chain_count == 0:
        raise ValueError("reward config cannot be reported with zero training-enabled chains")
    training_promotion = _validate_training_promotion(
        continuity.get("training_promotion"),
        training_chain_count=training_chain_count,
    )
    if configured_mode == "reward":
        if configured_coefficient <= 0.0:
            raise ValueError("reward config in joint report requires a positive coefficient")
        expected_calibration = _require_sha256(
            continuity_config.get("expected_calibration_fingerprint"),
            "resolved config calibration fingerprint",
        )
        if training_promotion is None:
            raise ValueError("reward joint report lacks continuity training-promotion evidence")
        _same_sha256(
            expected_calibration,
            training_promotion["calibration_fingerprint"],
            "continuity promotion calibration",
        )
        if configured_coefficient != training_promotion["selected_reward_coefficient"]:
            raise ValueError("resolved reward coefficient differs from continuity promotion evidence")

    rollout = copy.deepcopy(rollout_evidence["metrics"])
    task = _section_or_filtered(
        rollout,
        "task",
        ("return", "success", "termination", "coverage", "impact", "landing", "hit"),
    )
    tracking = _section_or_filtered(
        rollout,
        "tracking",
        ("err_", "tracking", "qpos", "qvel", "rpos", "rquat", "root_"),
    )
    online_synergy = _prefixed_scalars(rollout, "synergy_")
    online_continuity = _prefixed_scalars(rollout, "fascicle_continuity_")

    limitations = [
        "Kinematic tracking does not uniquely identify physiological muscle recruitment.",
        "Local fascicle continuity, low-dimensional control synergy, and EMG observation mapping are distinct hypotheses.",
        "The report aggregates source metrics and does not create new causal or neural-synergy evidence.",
    ]
    if training_chain_count == 0:
        limitations.append(
            "The bound continuity graph has no training-enabled chain; continuity results are diagnostic-only."
        )
    limitations.extend(emg_limitations)

    payload: dict[str, Any] = {
        "schema_version": JOINT_REPORT_SCHEMA_VERSION,
        "identity": {
            "branch_commit_sha": commit,
            "run_id": run_id,
            "config_hash": declared_config_hash,
            "resolved_config_fingerprint": source_fingerprints["resolved_config"],
            "checkpoint_fingerprint": checkpoint,
            "promotion_fingerprint": promotion,
            "taxonomy_fingerprint": taxonomy_fingerprint,
            "continuity_graph_fingerprint": continuity_fingerprint,
            "continuity_source_graph_fingerprint": (
                None if training_promotion is None else training_promotion["source_graph_fingerprint"]
            ),
            "continuity_calibration_fingerprint": (
                None if training_promotion is None else training_promotion["calibration_fingerprint"]
            ),
            "continuity_review_fingerprint": (
                None if training_promotion is None else training_promotion["review_fingerprint"]
            ),
            "formal_synergy_basis_fingerprint": formal_basis,
            "frozen_decoder_fingerprint": decoder_fingerprint,
            "rollout_metrics_fingerprint": rollout_evidence["artifact_fingerprint"],
            "emg_mapping_fingerprint": emg_identity.get("mapping_fingerprint"),
            "source_artifact_fingerprints": source_fingerprints,
            "data_evidence": {
                "source_dataset_fingerprint": dataset_fingerprint,
                "split_provenance_fingerprint": split_fingerprint,
                "event_reference_fingerprint": event_reference,
                "session_uid": session_uid,
                "evaluation_split_fingerprint": rollout_identity["evaluation_split_fingerprint"],
                "environment_fingerprint": rollout_identity["environment_fingerprint"],
                **emg_identity,
            },
        },
        "task": task,
        "tracking": tracking,
        "control_synergy": {
            "formal_basis_fingerprint": formal_basis,
            "basis_manifest": copy.deepcopy(dict(synergy_basis_manifest)),
            "frozen_decoder_fingerprint": decoder_fingerprint,
            "decoder_manifest": copy.deepcopy(dict(frozen_decoder_manifest)),
            "online_diagnostics": online_synergy,
        },
        "activation_consistency": {
            "signal_priority": copy.deepcopy(continuity.get("signal_priority")),
            "graph_fingerprint": continuity_fingerprint,
            "taxonomy_fingerprint": taxonomy_fingerprint,
            "coverage": copy.deepcopy(dict(continuity_coverage)),
            "activation": copy.deepcopy(continuity.get("activation")),
            "excitation": copy.deepcopy(continuity.get("excitation")),
            "online_diagnostics": online_continuity,
            "online_mode": configured_mode,
            "reward_coefficient": configured_coefficient,
            "training_promotion": training_promotion,
        },
        "mapped_15ch_synergy": mapped_synergy,
        "emg_comparison": emg_comparison,
        "claim_scope": {
            "continuity": (
                "verified_training_prior_and_diagnostic"
                if training_chain_count > 0 and configured_mode == "reward"
                else "simulation_activation_diagnostic_only"
            ),
            "control_synergy": "frozen_policy_action_representation",
            "emg": emg_identity.get("claim_scope", "not_measured"),
            "cross_space_causal_claim": False,
        },
        "limitations": limitations,
    }
    payload["report_fingerprint"] = joint_report_fingerprint(payload)
    return payload


def _bind_emg(
    report: Mapping[str, Any] | None,
    *,
    checkpoint: str,
    promotion: str,
    formal_basis: str,
    taxonomy_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    if report is None:
        return (
            {"mapping_fingerprint": None, "claim_scope": "not_measured"},
            {"status": "not_provided"},
            {"status": "not_provided"},
            ["No EMG report was supplied; no simulation-to-human synergy claim is available."],
        )
    policy = _mapping(report.get("policy_evidence"), "emg_report.policy_evidence")
    _same_sha256(checkpoint, policy.get("policy_checkpoint_fingerprint"), "EMG checkpoint")
    _same_sha256(promotion, policy.get("policy_promotion_fingerprint"), "EMG promotion")
    _same_sha256(formal_basis, policy.get("formal_synergy_basis_fingerprint"), "EMG formal basis")
    inputs = _mapping(report.get("input_fingerprints"), "emg_report.input_fingerprints")
    mapping_fingerprint = _require_sha256(
        inputs.get("mapping_json_sha256"),
        "EMG mapping JSON fingerprint",
    )
    mapping = _mapping(report.get("mapping"), "emg_report.mapping")
    model_binding = mapping.get("model_binding")
    if isinstance(model_binding, Mapping) and model_binding.get("taxonomy_fingerprint"):
        _same_sha256(
            taxonomy_fingerprint,
            model_binding["taxonomy_fingerprint"],
            "EMG mapping taxonomy",
        )
    trial = report.get("trial_binding")
    cohort = report.get("cohort_contract")
    evidence = trial if isinstance(trial, Mapping) else cohort
    evidence_fingerprint = None
    sessions: Any = None
    references: Any = None
    if isinstance(evidence, Mapping):
        evidence_fingerprint = _first_present(
            evidence,
            ("binding_fingerprint", "cohort_fingerprint"),
        )
        if evidence_fingerprint is not None:
            evidence_fingerprint = _require_sha256(
                evidence_fingerprint,
                "EMG trial/cohort binding fingerprint",
            )
        sessions = evidence.get("session_uids")
        references = evidence.get("reference_trial_fingerprints")
    mapped = report.get("synergy")
    if not isinstance(mapped, Mapping):
        mapped = report.get("nmf")
    if not isinstance(mapped, Mapping):
        mapped = {"status": "not_available_in_emg_report"}
    comparison = copy.deepcopy(dict(report))
    return (
        {
            "mapping_fingerprint": mapping_fingerprint,
            "emg_binding_fingerprint": evidence_fingerprint,
            "emg_session_uids": copy.deepcopy(sessions),
            "emg_reference_fingerprints": copy.deepcopy(references),
            "claim_scope": str(report.get("claim_scope", "unspecified")),
        },
        comparison,
        copy.deepcopy(dict(mapped)),
        list(report.get("claim_limitations", ())),
    )


def _validate_training_promotion(
    value: Any,
    *,
    training_chain_count: int,
) -> dict[str, Any] | None:
    if training_chain_count == 0:
        if value is not None:
            raise ValueError("zero-chain continuity report cannot declare training promotion")
        return None
    promotion = _mapping(value, "fascicle_continuity.training_promotion")
    expected_fields = {
        "batch",
        "source_graph_fingerprint",
        "calibration_fingerprint",
        "review_fingerprint",
        "selected_reward_coefficient",
        "promoted_chain_ids",
    }
    if set(promotion) != expected_fields:
        raise ValueError("continuity training-promotion fields differ from contract")
    batch = _nonempty_text(promotion["batch"], "continuity promotion batch")
    chain_ids = promotion["promoted_chain_ids"]
    if (
        not isinstance(chain_ids, list)
        or len(chain_ids) != training_chain_count
        or len(set(chain_ids)) != len(chain_ids)
        or any(not isinstance(item, str) or not item.strip() for item in chain_ids)
    ):
        raise ValueError("continuity promoted_chain_ids differ from training-chain coverage")
    coefficient = _finite_number(
        promotion["selected_reward_coefficient"],
        "continuity promoted reward coefficient",
    )
    if coefficient <= 0.0:
        raise ValueError("continuity promoted reward coefficient must be positive")
    return {
        "batch": batch,
        "source_graph_fingerprint": _require_sha256(
            promotion["source_graph_fingerprint"],
            "continuity promotion source graph fingerprint",
        ),
        "calibration_fingerprint": _require_sha256(
            promotion["calibration_fingerprint"],
            "continuity promotion calibration fingerprint",
        ),
        "review_fingerprint": _require_sha256(
            promotion["review_fingerprint"],
            "continuity promotion review fingerprint",
        ),
        "selected_reward_coefficient": coefficient,
        "promoted_chain_ids": [item.strip() for item in chain_ids],
    }


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return dict(value)


def _nested_mapping(value: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return {} if not isinstance(current, Mapping) else dict(current)


def _nonempty_text(value: Any, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field} must be non-empty")
    return result


def _require_sha256(value: Any, field: str) -> str:
    result = str(value or "").strip().lower()
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{field} must be lowercase 64-hex")
    return result


def _first_present(value: Mapping[str, Any], keys: Sequence[str]) -> Any:
    return next((value[key] for key in keys if value.get(key) not in (None, "")), None)


def _sha256_from(value: Mapping[str, Any], keys: Sequence[str], field: str) -> str:
    return _require_sha256(_first_present(value, keys), field)


def _same_sha256(expected: str, value: Any, field: str) -> None:
    if _require_sha256(value, field) != expected:
        raise ValueError(f"{field} differs from the selected joint-report evidence")


def _declared_or_canonical_fingerprint(
    value: Mapping[str, Any],
    declared_fields: Sequence[str],
    field: str,
) -> str:
    supplied = _first_present(value, declared_fields)
    if supplied is None:
        return _canonical_json_sha256(value)
    supplied_sha = _require_sha256(supplied, f"{field} fingerprint")
    unsigned = {key: item for key, item in value.items() if key not in declared_fields}
    if supplied_sha != _canonical_json_sha256(unsigned):
        raise ValueError(f"{field} declared fingerprint is stale")
    return supplied_sha


def _split_fingerprint(manifest: Mapping[str, Any]) -> str:
    explicit = _first_present(
        manifest,
        ("split_provenance_fingerprint", "data_split_fingerprint"),
    )
    if explicit is not None:
        return _require_sha256(explicit, "split provenance fingerprint")
    split = manifest.get("split_provenance")
    if not isinstance(split, Mapping):
        raise ValueError("synergy basis manifest has no split provenance binding")
    return _canonical_json_sha256(split)


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    result = int(value)
    if result < 0 or result != float(value):
        raise ValueError(f"{field} must be a non-negative integer")
    return result


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if result != result or result in (float("inf"), -float("inf")):
        raise ValueError(f"{field} must be finite")
    return result


def _numeric_metric_tree(value: Any, field: str) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _numeric_metric_tree(child, f"{field}.{key}") for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_numeric_metric_tree(child, f"{field}[{index}]") for index, child in enumerate(value)]
    return _finite_number(value, field)


def _section_or_filtered(
    rollout: Mapping[str, Any],
    section: str,
    fragments: Sequence[str],
) -> dict[str, Any]:
    nested = rollout.get(section)
    if isinstance(nested, Mapping):
        return copy.deepcopy(dict(nested))
    return {
        str(key): copy.deepcopy(value)
        for key, value in rollout.items()
        if any(fragment in str(key).lower() for fragment in fragments)
    }


def _prefixed_scalars(value: Mapping[str, Any], prefix: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, item in value.items():
        if not str(key).startswith(prefix):
            continue
        try:
            result[str(key)] = _finite_number(item, str(key))
        except (TypeError, ValueError):
            continue
    return result


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
    )
    return _mapping(payload, str(path))


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-metrics-json", type=Path, required=True)
    parser.add_argument("--physiology-report-json", type=Path, required=True)
    parser.add_argument("--synergy-basis-manifest-json", type=Path, required=True)
    parser.add_argument("--frozen-decoder-manifest-json", type=Path, required=True)
    parser.add_argument("--resolved-config-json", type=Path, required=True)
    parser.add_argument("--checkpoint-evidence-json", type=Path, required=True)
    parser.add_argument("--branch-commit-sha", required=True)
    parser.add_argument("--emg-report-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_joint_report(
        rollout_metrics=_load_json(args.rollout_metrics_json),
        physiology_report=_load_json(args.physiology_report_json),
        synergy_basis_manifest=_load_json(args.synergy_basis_manifest_json),
        frozen_decoder_manifest=_load_json(args.frozen_decoder_manifest_json),
        resolved_config=_load_json(args.resolved_config_json),
        checkpoint_evidence=_load_json(args.checkpoint_evidence_json),
        branch_commit_sha=args.branch_commit_sha,
        emg_report=(None if args.emg_report_json is None else _load_json(args.emg_report_json)),
    )
    _atomic_write_json(args.output_json, report)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
