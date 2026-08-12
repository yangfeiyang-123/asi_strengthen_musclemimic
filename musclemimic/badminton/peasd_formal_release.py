"""Build the fail-closed final PEASD formal-release report.

This module is the last evidence boundary in the PEASD workflow.  It does not
train a policy and it does not add a result-dependent performance threshold.
Instead, it revalidates the immutable Stage-1/2/3 gates, proves that they all
belong to one action and one upstream lineage, and requires a complete formal
evaluation artifact containing every metric needed to delimit the claim.

Stage 3 follows applicability rather than current asset readiness.  In
particular, Forehand Lift is a hitting action and therefore cannot be released
without a passed Stage-3 PEASD family gate.  ChinaJump is body-only and must be
declared explicitly not applicable; absence is never interpreted as success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from musclemimic.badminton.action_registry import action_choices, resolve
from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.badminton.stage1_peasd_gate import (
    validate_stage1_peasd_teacher_promotion,
)
from musclemimic.badminton.stage2_context_family import (
    validate_stage2_context_family_gate,
    validate_stage2_context_family_index,
    validate_stage2_shared_inputs,
)
from musclemimic.badminton.stage3_peasd_family import (
    validate_stage3_peasd_family_gate,
    validate_stage3_peasd_family_index,
)

FORMAL_RELEASE_SCHEMA_VERSION = "peasd_formal_release_report_v1"
COMPLETE_EVALUATION_SCHEMA_VERSION = "peasd_complete_evaluation_evidence_v1"

EXACT_TRAINING_SEEDS = (0, 1, 2)

COMMON_REQUIRED_METRIC_PATHS = (
    "physiology.m_channel.anchor_loss_by_channel",
    "physiology.m_channel.correlation_by_channel",
    "physiology.m_channel.peak_phase_error_by_channel",
    "physiology.m_channel.onset_error_s_by_channel",
    "physiology.m_channel.co_contraction_by_pair",
    "physiology.action_activation.action_rate",
    "physiology.action_activation.activation_rate",
    "physiology.action_activation.activation_energy",
    "physiology.action_activation.action_saturation_fraction",
    "physiology.action_activation.activation_saturation_fraction",
    "tracking_safety.fall_rate",
    "tracking_safety.early_termination_rate",
    "tracking_safety.joint_position_error_m",
    "tracking_safety.keypoint_position_error_m",
    "tracking_safety.root_position_error_m",
    "tracking_safety.tracking_error_m",
    "latent.context_response_l2",
    "latent.blank_context_response_l2",
    "latent.shuffled_context_head_loss",
    "latent.synergy_head_loss",
    "latent.synergy_head_correlation",
)

STAGE3_REQUIRED_METRIC_PATHS = (
    "stage3.hit_rate",
    "stage3.no_fall_rate",
    "stage3.opponent_back_landing_rate",
    "stage3.impact_position_error_m",
    "stage3.legal_landing_rate",
    "stage3.recovery_complete_rate",
    "stage3.normalized_control_energy",
)

_RATE_METRIC_PATHS = {
    "physiology.action_activation.action_saturation_fraction",
    "physiology.action_activation.activation_saturation_fraction",
    "tracking_safety.fall_rate",
    "tracking_safety.early_termination_rate",
    "stage3.hit_rate",
    "stage3.no_fall_rate",
    "stage3.opponent_back_landing_rate",
    "stage3.legal_landing_rate",
    "stage3.recovery_complete_rate",
}

_CORRELATION_METRIC_PATHS = {
    "latent.synergy_head_correlation",
}

_NONNEGATIVE_METRIC_PATHS = (
    set(COMMON_REQUIRED_METRIC_PATHS)
    | set(STAGE3_REQUIRED_METRIC_PATHS)
) - _RATE_METRIC_PATHS - _CORRELATION_METRIC_PATHS

_COMMON_METRIC_SOURCE_IDS = ("physiology", "stage1", "stage2")
_STAGE3_METRIC_SOURCE_ID = "stage3"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 fingerprint")
    return value


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _load_object(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        value = load_json_strict(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable strict JSON: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return source, value


def _path_record(
    path: str | Path,
    *,
    binding_sha256: str,
    schema_version: str,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    return {
        "path": str(source),
        "content_sha256": _file_sha256(source),
        "schema_version": str(schema_version),
        "binding_sha256": _require_sha256(
            binding_sha256, label=f"{source.name} binding_sha256"
        ),
    }


def _resolve_record_path(record: Mapping[str, Any], *, label: str) -> Path:
    try:
        return Path(str(record["path"])).expanduser().resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise ValueError(f"{label} has no resolvable source path") from exc


def _verify_path_record(
    record: Mapping[str, Any],
    *,
    expected_path: str | Path,
    expected_binding: str,
    label: str,
) -> None:
    path = Path(expected_path).expanduser().resolve(strict=True)
    if _resolve_record_path(record, label=label) != path:
        raise ValueError(f"{label} points to a different source")
    if record.get("content_sha256") != _file_sha256(path):
        raise ValueError(f"{label} content changed after it was sealed")
    recorded_binding = record.get("artifact_fingerprint", record.get("binding_sha256"))
    if recorded_binding != expected_binding:
        raise ValueError(f"{label} binds a different artifact fingerprint")


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite numeric value")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite numeric value") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _reject_nonfinite_tree(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite_tree(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_tree(item, label=f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be finite")


def _reject_forbidden_execution_markers(value: Any, *, label: str) -> None:
    """Reject hidden dry-run/placeholder/failure markers in evidence extras."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in {"dry_run", "is_dry_run"} and item is not False:
                raise ValueError(
                    f"{label}.{key} marks dry-run, placeholder, failed, or incomplete evidence"
                )
            if normalized in {"placeholder", "is_placeholder"} and item is not False:
                raise ValueError(
                    f"{label}.{key} marks dry-run, placeholder, failed, or incomplete evidence"
                )
            if normalized in {"failed", "is_failed", "failure"} and item is not False:
                raise ValueError(
                    f"{label}.{key} marks dry-run, placeholder, failed, or incomplete evidence"
                )
            if normalized in {"status", "result", "state"} and isinstance(item, str):
                if item.strip().lower().replace("-", "_") in {
                    "dry_run",
                    "placeholder",
                    "failed",
                    "failure",
                    "incomplete",
                }:
                    raise ValueError(
                        f"{label}.{key} marks dry-run, placeholder, failed, or incomplete evidence"
                    )
            _reject_forbidden_execution_markers(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_execution_markers(item, label=f"{label}[{index}]")
    elif isinstance(value, str) and value.strip().lower().replace("-", "_") in {
        "dry_run",
        "placeholder",
        "failed",
        "failure",
        "incomplete",
    }:
        raise ValueError(
            f"{label} marks dry-run, placeholder, failed, or incomplete evidence"
        )


def _metric_at(metrics: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = metrics
    for component in dotted_path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"complete evaluation is missing required metric {dotted_path}")
        value = value[component]
    return value


def _expected_metric_source_id(dotted_path: str) -> str:
    if dotted_path.startswith("physiology.m_channel."):
        return "physiology"
    if dotted_path.startswith("physiology.action_activation.") or dotted_path.startswith(
        "tracking_safety."
    ):
        return "stage1"
    if dotted_path.startswith("latent."):
        return "stage2"
    if dotted_path.startswith("stage3."):
        return _STAGE3_METRIC_SOURCE_ID
    raise ValueError(f"no immutable metric-source class for {dotted_path}")


def _validate_metric_source_provenance(
    *,
    evidence_path: Path,
    payload: Mapping[str, Any],
    metrics: Mapping[str, Any],
    required_paths: Sequence[str],
    stage3_applicable: bool,
) -> None:
    """Require every reported metric to be copied exactly from a bound source.

    A self-hash only proves that one JSON file is internally consistent.  It
    does not prove that its numeric values came from an evaluator.  Formal
    evidence therefore carries one immutable source record per evaluation
    layer plus an exact JSON-path selector for every required metric.
    """

    source_artifacts = _require_mapping(
        payload.get("source_artifacts"), label="evaluation source_artifacts"
    )
    expected_source_ids = set(_COMMON_METRIC_SOURCE_IDS)
    if stage3_applicable:
        expected_source_ids.add(_STAGE3_METRIC_SOURCE_ID)
    if set(source_artifacts) != expected_source_ids:
        raise ValueError(
            "complete evaluation must bind exact physiology/Stage1/Stage2"
            + ("/Stage3" if stage3_applicable else "")
            + " metric sources"
        )

    loaded_sources: dict[str, Mapping[str, Any]] = {}
    resolved_paths: list[Path] = []
    for source_id in sorted(expected_source_ids):
        record = _require_mapping(
            source_artifacts[source_id],
            label=f"evaluation source_artifacts.{source_id}",
        )
        source_path = _resolve_record_path(
            record, label=f"evaluation source_artifacts.{source_id}"
        )
        if source_path == evidence_path:
            raise ValueError("complete evaluation cannot cite itself as a metric source")
        if record.get("content_sha256") != _file_sha256(source_path):
            raise ValueError(f"{source_id} metric source content changed")
        source_payload = load_json_strict(source_path)
        if not isinstance(source_payload, Mapping):
            raise ValueError(f"{source_id} metric source must be a JSON object")
        schema = source_payload.get("schema_version")
        if (
            not isinstance(schema, str)
            or not schema.strip()
            or schema
            in {COMPLETE_EVALUATION_SCHEMA_VERSION, FORMAL_RELEASE_SCHEMA_VERSION}
            or record.get("schema_version") != schema
        ):
            raise ValueError(f"{source_id} metric source has an invalid schema binding")
        if "binding_sha256" in source_payload and record.get(
            "artifact_fingerprint"
        ) != source_payload.get("binding_sha256"):
            raise ValueError(f"{source_id} metric source binds another artifact")
        _reject_nonfinite_tree(source_payload, label=f"metric_source.{source_id}")
        _reject_forbidden_execution_markers(
            source_payload, label=f"metric_source.{source_id}"
        )
        loaded_sources[source_id] = source_payload
        resolved_paths.append(source_path)
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("complete evaluation metric sources must be distinct files")

    provenance = _require_mapping(
        payload.get("metric_provenance"), label="evaluation metric_provenance"
    )
    if set(provenance) != set(required_paths):
        raise ValueError(
            "complete evaluation metric_provenance must cover every required metric exactly"
        )
    for dotted_path in required_paths:
        record = _require_mapping(
            provenance[dotted_path],
            label=f"evaluation metric_provenance.{dotted_path}",
        )
        if set(record) != {"source", "json_path", "value_sha256"}:
            raise ValueError(f"metric provenance for {dotted_path} has unknown fields")
        expected_source = _expected_metric_source_id(dotted_path)
        if record.get("source") != expected_source:
            raise ValueError(f"metric {dotted_path} cites the wrong evaluator layer")
        json_path = record.get("json_path")
        if not isinstance(json_path, str) or not json_path:
            raise ValueError(f"metric {dotted_path} has no source JSON path")
        source_value = _metric_at(loaded_sources[expected_source], json_path)
        reported_value = _metric_at(metrics, dotted_path)
        if source_value != reported_value:
            raise ValueError(
                f"metric {dotted_path} does not match its immutable source artifact"
            )
        if record.get("value_sha256") != _canonical_value_sha256(source_value):
            raise ValueError(f"metric {dotted_path} source-value binding is stale")


def _validate_scalar_metric(metrics: Mapping[str, Any], dotted_path: str) -> None:
    value = _finite_number(_metric_at(metrics, dotted_path), label=dotted_path)
    if dotted_path in _RATE_METRIC_PATHS and not 0.0 <= value <= 1.0:
        raise ValueError(f"{dotted_path} must lie in [0,1]")
    if dotted_path in _CORRELATION_METRIC_PATHS and not -1.0 <= value <= 1.0:
        raise ValueError(f"{dotted_path} must lie in [-1,1]")
    if dotted_path in _NONNEGATIVE_METRIC_PATHS and value < 0.0:
        raise ValueError(f"{dotted_path} must be non-negative")


def _validate_channel_metrics(metrics: Mapping[str, Any]) -> None:
    physiology = _require_mapping(metrics.get("physiology"), label="metrics.physiology")
    channels = _require_mapping(
        physiology.get("m_channel"), label="metrics.physiology.m_channel"
    )
    channel_ids = channels.get("channel_ids")
    if (
        not isinstance(channel_ids, list)
        or not channel_ids
        or any(not isinstance(item, str) or not item.strip() for item in channel_ids)
        or len(set(channel_ids)) != len(channel_ids)
    ):
        raise ValueError("M-channel evidence requires non-empty unique channel_ids")
    if channels.get("channel_count") != len(channel_ids):
        raise ValueError("M-channel channel_count does not match channel_ids")
    expected_ids = set(channel_ids)
    for field in (
        "anchor_loss_by_channel",
        "correlation_by_channel",
        "peak_phase_error_by_channel",
        "onset_error_s_by_channel",
    ):
        values = _require_mapping(
            channels.get(field), label=f"metrics.physiology.m_channel.{field}"
        )
        if set(values) != expected_ids:
            raise ValueError(f"M-channel {field} must report every channel exactly once")
        for channel_id, raw_value in values.items():
            value = _finite_number(raw_value, label=f"{field}.{channel_id}")
            if field == "correlation_by_channel":
                if not -1.0 <= value <= 1.0:
                    raise ValueError(f"{field}.{channel_id} must lie in [-1,1]")
            elif value < 0.0:
                raise ValueError(f"{field}.{channel_id} must be non-negative")
    co_contraction = _require_mapping(
        channels.get("co_contraction_by_pair"),
        label="metrics.physiology.m_channel.co_contraction_by_pair",
    )
    if not co_contraction:
        raise ValueError("M-channel evidence must report at least one co-contraction pair")
    for pair, raw_value in co_contraction.items():
        if not isinstance(pair, str) or not pair.strip():
            raise ValueError("co-contraction pair names must be non-empty strings")
        if _finite_number(raw_value, label=f"co_contraction_by_pair.{pair}") < 0.0:
            raise ValueError(f"co_contraction_by_pair.{pair} must be non-negative")


def _expected_upstream_bindings(
    *,
    stage1_binding: str,
    stage2_binding: str,
    stage3_binding: str | None,
    stage3_applicable: bool,
) -> dict[str, Any]:
    stage3 = (
        {
            "status": "passed",
            "binding_sha256": _require_sha256(
                stage3_binding, label="Stage-3 family-gate binding"
            ),
        }
        if stage3_applicable
        else {
            "status": "not_applicable",
            "binding_sha256": None,
            "reason": "action_registry.stage3_applicable=false",
        }
    )
    return {
        "stage1_peasd_promotion_binding_sha256": _require_sha256(
            stage1_binding, label="Stage-1 promotion binding"
        ),
        "stage2_context_family_gate_binding_sha256": _require_sha256(
            stage2_binding, label="Stage-2 family-gate binding"
        ),
        "stage3_peasd_family_gate": stage3,
    }


def validate_complete_evaluation_evidence(
    source: str | Path,
    *,
    expected_action: str,
    expected_stage1_binding: str,
    expected_stage2_binding: str,
    expected_stage3_binding: str | None,
) -> dict[str, Any]:
    """Validate the self-bound, complete metric report used by final release."""

    evidence_path, payload = _load_object(
        source, label="complete PEASD evaluation evidence"
    )
    if payload.get("schema_version") != COMPLETE_EVALUATION_SCHEMA_VERSION:
        raise ValueError("unsupported complete PEASD evaluation-evidence schema")
    _reject_nonfinite_tree(payload, label="complete_evaluation")
    _reject_forbidden_execution_markers(payload, label="complete_evaluation")
    supplied = _require_sha256(
        payload.get("binding_sha256"), label="complete evaluation binding_sha256"
    )
    unsigned = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if supplied != _canonical_sha256(unsigned):
        raise ValueError("complete evaluation binding mismatch")

    spec = resolve(expected_action)
    if payload.get("action") != {"slug": spec.slug, "action_id": spec.action_id}:
        raise ValueError("complete evaluation belongs to another action")
    execution = _require_mapping(payload.get("execution"), label="evaluation execution")
    if execution != {
        "mode": "formal",
        "completed": True,
        "passed": True,
        "dry_run": False,
        "placeholder": False,
    }:
        raise ValueError(
            "complete evaluation must be a passed, completed formal run; "
            "dry-run, placeholder, failed, or incomplete evidence is forbidden"
        )

    expected_upstream = _expected_upstream_bindings(
        stage1_binding=expected_stage1_binding,
        stage2_binding=expected_stage2_binding,
        stage3_binding=expected_stage3_binding,
        stage3_applicable=spec.stage3_applicable,
    )
    if payload.get("upstream_bindings") != expected_upstream:
        raise ValueError("complete evaluation is bound to different upstream gates")

    statistical_scope = _require_mapping(
        payload.get("statistical_scope"), label="evaluation statistical_scope"
    )
    expected_statistical_scope = {
        "physiology_unit": "trial_subject_session",
        "rl_unit": "independent_training_seed",
        "rl_training_seeds": list(EXACT_TRAINING_SEEDS),
        "episode_frame_feed_as_independent_n": False,
        "significance_claimed": False,
        "population_level_effect_claimed": False,
        "population_physiology_claimed": False,
    }
    if statistical_scope != expected_statistical_scope:
        raise ValueError(
            "evaluation statistical scope must keep seed as the RL unit and must not "
            "claim significance or population-level inference"
        )

    metrics = _require_mapping(payload.get("metrics"), label="evaluation metrics")
    _validate_channel_metrics(metrics)
    required_paths = list(COMMON_REQUIRED_METRIC_PATHS)
    if spec.stage3_applicable:
        required_paths.extend(STAGE3_REQUIRED_METRIC_PATHS)
    elif "stage3" in metrics:
        raise ValueError("body-only evaluation must not report a Stage-3 hitting block")
    for dotted_path in required_paths:
        if ".m_channel." not in dotted_path:
            _validate_scalar_metric(metrics, dotted_path)
    _validate_metric_source_provenance(
        evidence_path=evidence_path,
        payload=payload,
        metrics=metrics,
        required_paths=required_paths,
        stage3_applicable=spec.stage3_applicable,
    )
    return payload


def _validate_stage2_lineage(
    *,
    stage2_family_gate_path: Path,
    stage1_promotion_path: Path,
    stage1_promotion: Mapping[str, Any],
    action: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    gate = validate_stage2_context_family_gate(
        stage2_family_gate_path, require_pass=True
    )
    if gate.get("passed") is not True:
        raise ValueError("Stage-2 context family gate did not pass")
    gate_index_record = _require_mapping(
        gate.get("family_index"), label="Stage-2 gate family_index"
    )
    index_path = _resolve_record_path(
        gate_index_record, label="Stage-2 gate family_index"
    )
    index = validate_stage2_context_family_index(index_path)
    _verify_path_record(
        gate_index_record,
        expected_path=index_path,
        expected_binding=str(index.get("binding_sha256", "")),
        label="Stage-2 gate family_index",
    )
    spec = resolve(action)
    expected_action = {
        "slug": spec.slug,
        "action_id": spec.action_id,
        "tube_action_id": spec.emg_trial_actions[0],
    }
    if index.get("action") != expected_action:
        raise ValueError("Stage-2 context family belongs to another action")

    shared_record = _require_mapping(
        index.get("shared_inputs"), label="Stage-2 index shared_inputs"
    )
    shared_path = _resolve_record_path(
        shared_record, label="Stage-2 index shared_inputs"
    )
    shared = validate_stage2_shared_inputs(shared_path, expected_action=spec.slug)
    _verify_path_record(
        shared_record,
        expected_path=shared_path,
        expected_binding=str(shared.get("binding_sha256", "")),
        label="Stage-2 index shared_inputs",
    )
    stage1_lineage = _require_mapping(
        shared.get("stage1_peasd"), label="Stage-2 shared Stage-1 lineage"
    )
    if stage1_lineage.get("promotion_binding_sha256") != stage1_promotion.get(
        "binding_sha256"
    ):
        raise ValueError("Stage-2 shared inputs bind a different Stage-1 promotion")
    stage1_record = _require_mapping(
        stage1_lineage.get("promotion"),
        label="Stage-2 shared Stage-1 promotion",
    )
    _verify_path_record(
        stage1_record,
        expected_path=stage1_promotion_path,
        expected_binding=str(stage1_promotion.get("binding_sha256", "")),
        label="Stage-2 shared Stage-1 promotion",
    )
    return gate, index, shared


def _validate_stage3_lineage(
    *,
    stage3_family_gate_path: Path,
    stage2_family_gate_path: Path,
    stage2_family_gate: Mapping[str, Any],
    action: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = validate_stage3_peasd_family_gate(
        stage3_family_gate_path, require_pass=True
    )
    if gate.get("passed") is not True:
        raise ValueError("Stage-3 PEASD family gate did not pass")
    index_record = _require_mapping(
        gate.get("family_index"), label="Stage-3 gate family_index"
    )
    index_path = _resolve_record_path(index_record, label="Stage-3 gate family_index")
    index = validate_stage3_peasd_family_index(index_path)
    _verify_path_record(
        index_record,
        expected_path=index_path,
        expected_binding=str(index.get("binding_sha256", "")),
        label="Stage-3 gate family_index",
    )
    spec = resolve(action)
    action_identity = _require_mapping(
        index.get("action"), label="Stage-3 family action"
    )
    if (
        action_identity.get("slug") != spec.slug
        or action_identity.get("action_id") != spec.action_id
    ):
        raise ValueError("Stage-3 PEASD family belongs to another action")
    stage2_record = _require_mapping(
        _require_mapping(
            index.get("stage2_context_family"),
            label="Stage-3 family Stage-2 lineage",
        ).get("gate"),
        label="Stage-3 family Stage-2 gate",
    )
    _verify_path_record(
        stage2_record,
        expected_path=stage2_family_gate_path,
        expected_binding=str(stage2_family_gate.get("binding_sha256", "")),
        label="Stage-3 family Stage-2 gate",
    )
    return gate, index


def _claim_scope(*, stage3_applicable: bool) -> dict[str, Any]:
    supported = [
        "action-specific Stage-1 PEASD physiology-guided teacher",
        "action-specific Stage-2 privileged-context family comparison",
        "reported action, activation, safety, tracking, and latent diagnostics",
    ]
    excluded = [
        "episode/frame/feed counts as independent statistical replicates",
        "null-hypothesis significance from three training seeds",
        "population-level physiology or policy effects",
        "unreported or proxy metrics as substitutes for required evidence",
    ]
    if stage3_applicable:
        supported.append("Stage-3 racket-hit PEASD family under the sealed task protocol")
    else:
        supported.append("body-only action/control scope")
        excluded.extend(
            [
                "racket-control claim",
                "shuttle-hit or return claim",
                "Stage-3 claim",
            ]
        )
    return {"supported": supported, "excluded": excluded}


def build_peasd_formal_release_report(
    *,
    action: str,
    stage1_peasd_promotion: str | Path,
    stage2_context_family_gate: str | Path,
    complete_evaluation_evidence: str | Path,
    stage3_peasd_family_gate: str | Path | None = None,
    stage3_not_applicable: bool = False,
) -> dict[str, Any]:
    """Build one deterministic release report after rebuilding every input."""

    spec = resolve(action)
    stage1_path = Path(stage1_peasd_promotion).expanduser().resolve(strict=True)
    stage2_path = Path(stage2_context_family_gate).expanduser().resolve(strict=True)
    evaluation_path = Path(complete_evaluation_evidence).expanduser().resolve(
        strict=True
    )

    stage1 = validate_stage1_peasd_teacher_promotion(
        stage1_path, expected_action=spec.slug
    )
    if stage1.get("passed") is not True:
        raise ValueError("Stage-1 PEASD teacher promotion did not pass")
    if (
        stage1.get("action") != spec.slug
        or stage1.get("action_id") != spec.action_id
    ):
        raise ValueError("Stage-1 PEASD teacher promotion belongs to another action")
    stage1_binding = _require_sha256(
        stage1.get("binding_sha256"), label="Stage-1 promotion binding_sha256"
    )

    stage2, stage2_index, stage2_shared = _validate_stage2_lineage(
        stage2_family_gate_path=stage2_path,
        stage1_promotion_path=stage1_path,
        stage1_promotion=stage1,
        action=spec.slug,
    )
    stage2_binding = _require_sha256(
        stage2.get("binding_sha256"), label="Stage-2 family-gate binding_sha256"
    )

    stage3_path: Path | None = None
    stage3: dict[str, Any] | None = None
    stage3_index: dict[str, Any] | None = None
    if spec.stage3_applicable:
        if stage3_not_applicable:
            raise ValueError(
                f"{spec.slug} is scientifically Stage-3 applicable; N/A cannot release it"
            )
        if stage3_peasd_family_gate is None:
            raise ValueError(
                f"{spec.slug} is scientifically Stage-3 applicable and requires a passed "
                "Stage-3 PEASD family gate"
            )
        stage3_path = Path(stage3_peasd_family_gate).expanduser().resolve(strict=True)
        stage3, stage3_index = _validate_stage3_lineage(
            stage3_family_gate_path=stage3_path,
            stage2_family_gate_path=stage2_path,
            stage2_family_gate=stage2,
            action=spec.slug,
        )
        stage3_binding: str | None = _require_sha256(
            stage3.get("binding_sha256"), label="Stage-3 family-gate binding_sha256"
        )
        stage3_disposition: dict[str, Any] = {
            "status": "passed",
            "applicable": True,
            "gate": _path_record(
                stage3_path,
                binding_sha256=stage3_binding,
                schema_version=str(stage3.get("schema_version", "")),
            ),
            "family_index": _path_record(
                _resolve_record_path(
                    _require_mapping(
                        stage3.get("family_index"),
                        label="Stage-3 gate family_index",
                    ),
                    label="Stage-3 gate family_index",
                ),
                binding_sha256=str(stage3_index.get("binding_sha256", "")),
                schema_version=str(stage3_index.get("schema_version", "")),
            ),
        }
    else:
        if stage3_peasd_family_gate is not None:
            raise ValueError(
                f"{spec.slug} is Stage-3-inapplicable and must not bind a hitting gate"
            )
        if not stage3_not_applicable:
            raise ValueError(
                f"{spec.slug} requires explicit --stage3-not-applicable; missing is not pass"
            )
        stage3_binding = None
        stage3_disposition = {
            "status": "not_applicable",
            "applicable": False,
            "gate": None,
            "reason": "action_registry.stage3_applicable=false",
            "missing_treated_as_pass": False,
        }

    evaluation = validate_complete_evaluation_evidence(
        evaluation_path,
        expected_action=spec.slug,
        expected_stage1_binding=stage1_binding,
        expected_stage2_binding=stage2_binding,
        expected_stage3_binding=stage3_binding,
    )
    evaluation_binding = _require_sha256(
        evaluation.get("binding_sha256"),
        label="complete evaluation binding_sha256",
    )

    payload: dict[str, Any] = {
        "schema_version": FORMAL_RELEASE_SCHEMA_VERSION,
        "action": {
            "slug": spec.slug,
            "action_id": spec.action_id,
            "racket_applicable": spec.racket_applicable,
            "stage3_applicable": spec.stage3_applicable,
        },
        "release_status": "passed",
        "passed": True,
        "formal": True,
        "dry_run": False,
        "placeholder": False,
        "upstream": {
            "stage1_peasd_promotion": _path_record(
                stage1_path,
                binding_sha256=stage1_binding,
                schema_version=str(stage1.get("schema_version", "")),
            ),
            "stage2_context_family_gate": _path_record(
                stage2_path,
                binding_sha256=stage2_binding,
                schema_version=str(stage2.get("schema_version", "")),
            ),
            "stage2_context_family_index": _path_record(
                _resolve_record_path(
                    _require_mapping(
                        stage2.get("family_index"),
                        label="Stage-2 gate family_index",
                    ),
                    label="Stage-2 gate family_index",
                ),
                binding_sha256=str(stage2_index.get("binding_sha256", "")),
                schema_version=str(stage2_index.get("schema_version", "")),
            ),
            "stage2_shared_inputs": _path_record(
                _resolve_record_path(
                    _require_mapping(
                        stage2_index.get("shared_inputs"),
                        label="Stage-2 index shared_inputs",
                    ),
                    label="Stage-2 index shared_inputs",
                ),
                binding_sha256=str(stage2_shared.get("binding_sha256", "")),
                schema_version=str(stage2_shared.get("schema_version", "")),
            ),
            "stage3_peasd_family": stage3_disposition,
        },
        "evaluation_evidence": _path_record(
            evaluation_path,
            binding_sha256=evaluation_binding,
            schema_version=str(evaluation.get("schema_version", "")),
        ),
        "required_reported_metric_paths": {
            "common": list(COMMON_REQUIRED_METRIC_PATHS),
            "stage3": (
                list(STAGE3_REQUIRED_METRIC_PATHS)
                if spec.stage3_applicable
                else {
                    "status": "not_applicable",
                    "reason": "action_registry.stage3_applicable=false",
                }
            ),
        },
        "reported_metrics": evaluation["metrics"],
        "statistical_scope": evaluation["statistical_scope"],
        "claim_scope": _claim_scope(stage3_applicable=spec.stage3_applicable),
        "acceptance_policy": {
            "only_pre_registered_upstream_gates_determine_acceptance": True,
            "release_added_numeric_thresholds": False,
            "reported_metrics_are_required_descriptive_evidence": True,
            "statistical_unit": "independent_training_seed",
            "episodes_frames_and_feeds_are_repeated_measurements": True,
            "significance_claimed": False,
        },
    }
    payload["binding_sha256"] = _canonical_sha256(payload)
    return payload


def validate_peasd_formal_release_report(
    source: str | Path,
    *,
    expected_action: str | None = None,
) -> dict[str, Any]:
    """Rebuild a final release from every bound source and compare exactly."""

    release_path, payload = _load_object(source, label="PEASD formal release")
    if payload.get("schema_version") != FORMAL_RELEASE_SCHEMA_VERSION:
        raise ValueError("unsupported PEASD formal-release schema")
    supplied = _require_sha256(
        payload.get("binding_sha256"), label="PEASD formal-release binding_sha256"
    )
    unsigned = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if supplied != _canonical_sha256(unsigned):
        raise ValueError("PEASD formal-release binding mismatch")
    if (
        payload.get("passed") is not True
        or payload.get("release_status") != "passed"
        or payload.get("formal") is not True
        or payload.get("dry_run") is not False
        or payload.get("placeholder") is not False
    ):
        raise ValueError("PEASD formal release is failed, incomplete, dry-run, or placeholder")

    action_record = _require_mapping(payload.get("action"), label="release action")
    spec = resolve(expected_action or str(action_record.get("slug", "")))
    expected_identity = {
        "slug": spec.slug,
        "action_id": spec.action_id,
        "racket_applicable": spec.racket_applicable,
        "stage3_applicable": spec.stage3_applicable,
    }
    if action_record != expected_identity:
        raise ValueError("PEASD formal release belongs to another action")
    upstream = _require_mapping(payload.get("upstream"), label="release upstream")
    stage1_record = _require_mapping(
        upstream.get("stage1_peasd_promotion"), label="release Stage-1 promotion"
    )
    stage2_record = _require_mapping(
        upstream.get("stage2_context_family_gate"), label="release Stage-2 gate"
    )
    stage3_disposition = _require_mapping(
        upstream.get("stage3_peasd_family"), label="release Stage-3 disposition"
    )
    evaluation_record = _require_mapping(
        payload.get("evaluation_evidence"), label="release evaluation evidence"
    )
    if spec.stage3_applicable:
        if (
            stage3_disposition.get("status") != "passed"
            or stage3_disposition.get("applicable") is not True
        ):
            raise ValueError("applicable Stage-3 release has no passed gate")
        stage3_record = _require_mapping(
            stage3_disposition.get("gate"), label="release Stage-3 gate"
        )
        stage3_path: Path | None = _resolve_record_path(
            stage3_record, label="release Stage-3 gate"
        )
        stage3_not_applicable = False
    else:
        expected_disposition = {
            "status": "not_applicable",
            "applicable": False,
            "gate": None,
            "reason": "action_registry.stage3_applicable=false",
            "missing_treated_as_pass": False,
        }
        if stage3_disposition != expected_disposition:
            raise ValueError("body-only release lacks the exact explicit Stage-3 N/A record")
        stage3_path = None
        stage3_not_applicable = True

    rebuilt = build_peasd_formal_release_report(
        action=spec.slug,
        stage1_peasd_promotion=_resolve_record_path(
            stage1_record, label="release Stage-1 promotion"
        ),
        stage2_context_family_gate=_resolve_record_path(
            stage2_record, label="release Stage-2 gate"
        ),
        stage3_peasd_family_gate=stage3_path,
        stage3_not_applicable=stage3_not_applicable,
        complete_evaluation_evidence=_resolve_record_path(
            evaluation_record, label="release evaluation evidence"
        ),
    )
    if rebuilt != payload:
        raise ValueError(
            f"PEASD formal release or one of its immutable sources changed: {release_path}"
        )
    return payload


def _write_immutable(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(payload), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(
                f"refusing to replace immutable PEASD formal-release artifact: {output}"
            )
        return output
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build the final immutable PEASD release")
    build.add_argument("--action", choices=action_choices(), required=True)
    build.add_argument("--stage1-peasd-promotion", type=Path, required=True)
    build.add_argument("--stage2-context-family-gate", type=Path, required=True)
    stage3 = build.add_mutually_exclusive_group(required=True)
    stage3.add_argument("--stage3-peasd-family-gate", type=Path)
    stage3.add_argument("--stage3-not-applicable", action="store_true")
    build.add_argument("--complete-evaluation-evidence", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    validate = sub.add_parser("validate", help="rebuild and validate a release")
    validate.add_argument("--release", type=Path, required=True)
    validate.add_argument("--expected-action", choices=action_choices())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        payload = build_peasd_formal_release_report(
            action=args.action,
            stage1_peasd_promotion=args.stage1_peasd_promotion,
            stage2_context_family_gate=args.stage2_context_family_gate,
            stage3_peasd_family_gate=args.stage3_peasd_family_gate,
            stage3_not_applicable=bool(args.stage3_not_applicable),
            complete_evaluation_evidence=args.complete_evaluation_evidence,
        )
        output = _write_immutable(args.output, payload)
        validate_peasd_formal_release_report(output, expected_action=args.action)
    else:
        output = Path(args.release)
        validate_peasd_formal_release_report(
            output, expected_action=args.expected_action
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
