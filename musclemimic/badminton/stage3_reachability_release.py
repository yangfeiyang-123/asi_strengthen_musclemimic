"""Seal the Stage-3 reachability chain before static PPO may start.

The Stage-3 search utilities intentionally emit separate artifacts: the CEM
report/candidate, an independent CPU audit, a cross-backend quality seal, and
the selected-correction BC checkpoint.  A directory name is not sufficient
provenance for that chain.  This module binds the exact bytes and identities in
two immutable manifests:

``stage3_successful_correction_dataset_manifest_v1``
    CEM report + final candidate -> standalone CPU audit -> cross-backend seal
    -> successful correction trajectory.

``stage3_reachability_release_v1``
    The successful correction manifest -> a pure short-BC checkpoint saved at
    zero PPO environment steps.

The final release authorizes only the next static/single-feed PPO step.  It is
not evidence that PPO, multi-feed generalization, or held-out return succeeded.
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

import numpy as np
import yaml

from musclemimic.badminton.action_registry import action_choices
from musclemimic.badminton.action_registry import resolve as resolve_action

REPO_ROOT = Path(__file__).resolve().parents[2]

CORRECTION_DATASET_SCHEMA = "stage3_successful_correction_dataset_manifest_v1"
REACHABILITY_RELEASE_SCHEMA = "stage3_reachability_release_v1"
STATIC_PPO_ENTRY_SCHEMA = "stage3_static_ppo_reachability_entry_v1"

NATIVE_CEM_REPORT_SCHEMA = "stage3_single_feed_mjx_cem_report_v3"
CEM_CONTRACT_SCHEMA = "stage3_single_feed_mjx_cem_v4"
CEM_CANDIDATE_SCHEMA = "stage3_cem_teacher_candidate_v1"
INLINE_CPU_AUDIT_SCHEMA = "stage3_cem_inline_cpu_quality_gate_v3"
CPU_AUDIT_SCHEMA = "stage3_cem_intermediate_cpu_quality_audit_v1"
CROSS_BACKEND_SCHEMA = "stage3_cross_backend_quality_teacher_report_v3"
QUALITY_TEACHER_BINDING_SCHEMA = "stage3_quality_teacher_dataset_binding_v1"
SHORT_BC_REPORT_SCHEMA = "stage3_selected_correction_bc_pretrain_v1"
VERSIONED_CHECKPOINT_SCHEMA = "incoming_hit_versioned_checkpoint_v1"
LATEST_POINTER_SCHEMA = "incoming_hit_checkpoint_pointer_v1"

PIPELINE_ARTIFACT_FIELDS = (
    "stage3_cem_contract",
    "stage3_cem_report",
    "stage3_cem_candidate",
    "stage3_cpu_audit_report",
    "stage3_cpu_audit_trace",
    "stage3_cross_backend_seal_report",
    "stage3_correction_dataset",
    "stage3_correction_dataset_manifest",
    "stage3_short_bc_checkpoint",
    "stage3_short_bc_metrics",
    "stage3_short_bc_train_report",
    "stage3_reachability_release",
)

PIPELINE_STEP_NAMES = (
    "stage3_single_feed_cem",
    "stage3_candidate_cpu_audit",
    "stage3_cross_backend_seal",
    "stage3_correction_dataset_seal",
    "stage3_short_bc",
    "stage3_reachability_release",
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
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


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 fingerprint")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a file: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {resolved}") from exc
    return resolved, _require_mapping(payload, label)


def _resolve_recorded_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is missing: {path}") from exc


def _bound_file(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"bound artifact must be a file: {resolved}")
    return {"path": str(resolved), "sha256": _file_sha256(resolved)}


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if output.exists():
        if output.is_file() and output.read_text(encoding="utf-8") == encoded:
            return output
        raise FileExistsError(
            f"refusing to replace an immutable Stage-3 release artifact: {output}"
        )
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, output)
    return output


def _parameter_sha256(parameters: Any) -> str:
    values = np.asarray(parameters, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("CEM candidate parameters must be a finite vector")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _verify_self_hash(
    payload: Mapping[str, Any],
    *,
    hash_key: str,
    label: str,
) -> None:
    recorded = payload.get(hash_key)
    unsigned = dict(payload)
    unsigned.pop(hash_key, None)
    if recorded != _canonical_sha256(unsigned):
        raise ValueError(f"{label} binding fingerprint mismatch")


def _resolve_checkpoint(path: str | Path) -> dict[str, Any]:
    """Resolve and validate one Stage-3 checkpoint without importing JAX."""

    requested = Path(path).expanduser()
    pointer_path: Path | None = None
    if requested.is_dir():
        if (requested / "_COMPLETE.json").is_file():
            payload_path = requested / "policy.npz"
        elif (requested / "policy_latest.json").is_file():
            requested = requested / "policy_latest.json"
        else:
            raise FileNotFoundError(f"checkpoint directory is incomplete: {requested}")
    if requested.suffix == ".json" and requested.is_file():
        pointer_path, pointer = _load_json(requested, "checkpoint pointer")
        if pointer.get("schema_version") != LATEST_POINTER_SCHEMA:
            raise ValueError("checkpoint pointer schema is incompatible")
        _verify_self_hash(
            pointer,
            hash_key="binding_sha256",
            label="checkpoint pointer",
        )
        payload_path = _resolve_recorded_path(
            pointer.get("payload_path"),
            "checkpoint pointer payload",
        )
        if pointer.get("payload_sha256") != _file_sha256(payload_path):
            raise ValueError("checkpoint pointer payload fingerprint mismatch")
    elif not requested.is_dir():
        try:
            payload_path = requested.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"checkpoint is missing: {requested}") from exc

    if not payload_path.is_file() or payload_path.suffix != ".npz":
        raise ValueError(f"checkpoint payload must be an NPZ file: {payload_path}")
    metadata_path = payload_path.with_suffix(".json")
    _, metadata = _load_json(metadata_path, "checkpoint metadata")
    payload_sha256 = _file_sha256(payload_path)
    metadata_sha256 = _file_sha256(metadata_path)
    if metadata.get("training_payload_sha256") != payload_sha256:
        raise ValueError("checkpoint metadata payload fingerprint mismatch")

    completion_path = payload_path.parent / "_COMPLETE.json"
    completion: dict[str, Any] | None = None
    if completion_path.is_file():
        _, completion = _load_json(completion_path, "checkpoint completion")
        if completion.get("schema_version") != VERSIONED_CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint completion schema is incompatible")
        _verify_self_hash(
            completion,
            hash_key="binding_sha256",
            label="checkpoint completion",
        )
        if (
            completion.get("payload_sha256") != payload_sha256
            or completion.get("metadata_sha256") != metadata_sha256
        ):
            raise ValueError("versioned checkpoint content fingerprint mismatch")
        if metadata.get("versioned_checkpoint_schema") != VERSIONED_CHECKPOINT_SCHEMA:
            raise ValueError("versioned checkpoint metadata schema is incompatible")
        if metadata.get("version_name") != completion.get("version_name"):
            raise ValueError("checkpoint metadata/completion version differs")

    if pointer_path is not None:
        _, pointer = _load_json(pointer_path, "checkpoint pointer")
        if pointer.get("metadata_sha256") != metadata_sha256:
            raise ValueError("checkpoint pointer metadata fingerprint mismatch")
        if completion is not None and pointer.get("version_name") != completion.get("version_name"):
            raise ValueError("checkpoint pointer/completion version differs")

    return {
        "requested_path": str(Path(path).expanduser().resolve(strict=True)),
        "payload_path": str(payload_path),
        "payload_sha256": payload_sha256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_sha256,
        "pointer_path": None if pointer_path is None else str(pointer_path),
        "pointer_sha256": None if pointer_path is None else _file_sha256(pointer_path),
        "completion_path": None if completion is None else str(completion_path),
        "completion_sha256": None if completion is None else _file_sha256(completion_path),
        "metadata": metadata,
        "versioned": completion is not None,
    }


def _latent_identity(
    control_manifest: Mapping[str, Any],
    expected_fingerprint: str | None,
) -> dict[str, Any]:
    actual = control_manifest.get("latent_checkpoint_fingerprint")
    if expected_fingerprint is None:
        if actual not in {None, ""}:
            raise ValueError("control unexpectedly uses a latent checkpoint")
        return {"kind": "explicit_no_latent", "fingerprint": None}
    expected = _require_sha256(
        expected_fingerprint,
        "expected latent checkpoint fingerprint",
    )
    if actual != expected:
        raise ValueError("control uses the wrong latent checkpoint fingerprint")
    return {"kind": "latent_checkpoint", "fingerprint": expected}


def _npz_scalar(payload: Mapping[str, Any], name: str) -> Any:
    if name not in payload:
        raise ValueError(f"trajectory is missing {name}")
    return np.asarray(payload[name]).item()


def _validate_contract_hash(contract: Mapping[str, Any]) -> str:
    if contract.get("schema_version") != CEM_CONTRACT_SCHEMA:
        raise ValueError("CEM contract schema is incompatible")
    recorded = _require_sha256(
        contract.get("contract_sha256"),
        "CEM contract fingerprint",
    )
    unsigned = dict(contract)
    unsigned.pop("contract_sha256", None)
    if recorded != _canonical_sha256(unsigned):
        raise ValueError("CEM contract fingerprint mismatch")
    return recorded


def _validate_policy_contract_hash(contract: Mapping[str, Any]) -> str:
    recorded = _require_sha256(
        contract.get("contract_sha256"),
        "policy-update contract fingerprint",
    )
    unsigned = dict(contract)
    unsigned.pop("contract_sha256", None)
    if recorded != _canonical_sha256(unsigned):
        raise ValueError("policy-update contract fingerprint mismatch")
    return recorded


def _validate_spec_and_scene(
    *,
    contract: Mapping[str, Any],
    expected_spec: str | Path,
) -> dict[str, Any]:
    expected_path = Path(expected_spec).expanduser()
    if not expected_path.is_absolute():
        expected_path = REPO_ROOT / expected_path
    expected_path = expected_path.resolve(strict=True)
    contract_spec = _resolve_recorded_path(contract.get("spec"), "CEM spec")
    if contract_spec != expected_path:
        raise ValueError("CEM report belongs to the wrong Stage-3 spec")
    try:
        spec = yaml.safe_load(expected_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("Stage-3 spec is unreadable") from exc
    spec = _require_mapping(spec, "Stage-3 spec")
    scene = _require_mapping(spec.get("scene"), "Stage-3 spec scene")
    scene_path = _resolve_recorded_path(scene.get("xml"), "Stage-3 scene")
    scene_sha256 = _file_sha256(scene_path)
    if contract.get("scene_sha256") != scene_sha256:
        raise ValueError("CEM scene fingerprint differs from the current Stage-3 spec")
    task_action = spec.get("action")
    experiment_id = spec.get("experiment_id")
    runner_type = spec.get("runner_type")
    stage3_v2 = _require_mapping(
        spec.get("stage3_v2"),
        "Stage-3 v2 task profile",
    )
    task_profile = stage3_v2.get("profile")
    if not isinstance(task_action, str) or not task_action:
        raise ValueError("Stage-3 spec has no action identity")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("Stage-3 spec has no experiment identity")
    if (
        runner_type != "incoming_shuttle_hit"
        or task_action != "IncomingShuttleHitImpactRecovery"
        or task_profile != "impact_recovery_v2"
    ):
        raise ValueError("Stage-3 spec has the wrong v2 task identity")
    return {
        "spec_path": str(expected_path),
        "spec_sha256": _file_sha256(expected_path),
        "stage3_runner_type": runner_type,
        "stage3_task_action": task_action,
        "stage3_task_profile": task_profile,
        "stage3_experiment_id": experiment_id,
        "scene_path": str(scene_path),
        "scene_sha256": scene_sha256,
    }


def _require_registered_stage3_v2_spec(
    *,
    action_spec: Any,
    expected_spec: str | Path,
) -> Path:
    """Require the exact action-owned Stage-3 v2 task asset from the registry."""

    registered = Path(action_spec.require("stage3_v2_spec")).expanduser()
    if not registered.is_absolute():
        registered = REPO_ROOT / registered
    registered = registered.resolve(strict=True)
    expected = Path(expected_spec).expanduser()
    if not expected.is_absolute():
        expected = REPO_ROOT / expected
    expected = expected.resolve(strict=True)
    if expected != registered:
        raise ValueError(
            f"action {action_spec.slug!r} requires its registered Stage-3 v2 "
            f"spec {registered}; got {expected}"
        )
    return registered


def _bounded_residual_treatment(
    control_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize the complete grouped residual ABI, including disabled arms."""

    control = _require_mapping(control_manifest, "Stage-3 control manifest")
    if control.get("schema_version") != "stage3_lab_control_v1":
        raise ValueError("bounded-residual treatment requires Stage-3 LAB control")
    raw_dimension = control.get("bounded_residual_dim")
    if type(raw_dimension) is not int or raw_dimension < 0:
        raise ValueError("Stage-3 control has an invalid bounded-residual dimension")
    dimension = raw_dimension
    schema_hash = control.get("bounded_residual_schema_hash")
    groups = control.get("bounded_residual_groups")
    if dimension == 0:
        if schema_hash is not None or groups is not None:
            raise ValueError(
                "disabled bounded residual must have null schema and groups"
            )
        return {
            "enabled": False,
            "dimension": 0,
            "schema_sha256": None,
            "groups": None,
        }

    fingerprint = _require_sha256(
        schema_hash,
        "bounded-residual schema fingerprint",
    )
    if not isinstance(groups, list) or not groups:
        raise ValueError("enabled bounded residual requires non-empty groups")
    normalized_groups: list[dict[str, Any]] = []
    seen_group_names: set[str] = set()
    seen_actuator_names: set[str] = set()
    total_dimension = 0
    for index, raw_group in enumerate(groups):
        group = _require_mapping(
            raw_group,
            f"bounded-residual group {index}",
        )
        if set(group) != {"name", "actuator_names", "alpha", "dim"}:
            raise ValueError(
                "bounded-residual groups require exact name/actuator_names/alpha/dim fields"
            )
        name = group.get("name")
        actuator_names = group.get("actuator_names")
        if not isinstance(name, str) or not name or name in seen_group_names:
            raise ValueError("bounded-residual group names must be non-empty and unique")
        if (
            not isinstance(actuator_names, list)
            or not actuator_names
            or any(not isinstance(value, str) or not value for value in actuator_names)
            or len(set(actuator_names)) != len(actuator_names)
        ):
            raise ValueError(
                "bounded-residual actuator_names must be a non-empty unique roster"
            )
        overlap = seen_actuator_names.intersection(actuator_names)
        if overlap:
            raise ValueError(
                "bounded-residual groups must have disjoint actuator_names"
            )
        alpha = _require_finite(
            group.get("alpha"),
            f"bounded-residual group {name} alpha",
        )
        group_dimension = group.get("dim")
        if (
            type(group_dimension) is not int
            or group_dimension != len(actuator_names)
            or not 0.0 <= alpha <= 0.10
        ):
            raise ValueError(
                "bounded-residual group dimension/alpha is incompatible"
            )
        seen_group_names.add(name)
        seen_actuator_names.update(actuator_names)
        total_dimension += group_dimension
        normalized_groups.append(
            {
                "name": name,
                "actuator_names": list(actuator_names),
                "alpha": alpha,
                "dim": group_dimension,
            }
        )
    if total_dimension != dimension:
        raise ValueError("bounded-residual groups do not match their total dimension")
    return {
        "enabled": True,
        "dimension": dimension,
        "schema_sha256": fingerprint,
        "groups": normalized_groups,
    }


def _validate_candidate(
    *,
    path: str | Path,
    contract: Mapping[str, Any],
    source_report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_path, candidate = _load_json(path, "CEM candidate")
    if candidate.get("schema_version") != CEM_CANDIDATE_SCHEMA:
        raise ValueError("CEM candidate is not a qualified teacher candidate")
    if candidate.get("contract_sha256") != contract.get("contract_sha256"):
        raise ValueError("CEM candidate is detached from its contract")
    parameters = np.asarray(candidate.get("parameters"), dtype=np.float32)
    expected_count = int(contract.get("parameter_count", 0))
    if parameters.shape != (expected_count,) or not np.isfinite(parameters).all():
        raise ValueError("CEM candidate parameter vector is incompatible")
    parameter_sha256 = _parameter_sha256(parameters)
    if candidate.get("metrics") != source_report.get("best_search_metrics"):
        raise ValueError("CEM candidate metrics differ from the final CEM report")
    if candidate.get("cpu_quality_audit") != source_report.get("cpu_gated_best_audit"):
        raise ValueError("CEM candidate CPU audit differs from the final CEM report")
    cpu_quality = _require_mapping(
        candidate.get("cpu_quality_audit"),
        "CEM candidate CPU quality audit",
    )
    if (
        cpu_quality.get("schema_version") != INLINE_CPU_AUDIT_SCHEMA
        or cpu_quality.get("cpu_quality_passed") is not True
        or cpu_quality.get("candidate_parameter_f32_sha256") != parameter_sha256
        or not np.array_equal(
            np.asarray(cpu_quality.get("candidate_parameters"), dtype=np.float32),
            parameters,
        )
    ):
        raise ValueError("CEM candidate lacks a matching passing inline CPU audit")
    return candidate, {
        **_bound_file(candidate_path),
        "schema_version": CEM_CANDIDATE_SCHEMA,
        "iteration": int(candidate.get("iteration", -1)),
        "parameter_f32_sha256": parameter_sha256,
        "parameter_count": int(parameters.size),
    }


def _validate_standalone_cpu_audit(
    *,
    path: str | Path,
    candidate_binding: Mapping[str, Any],
    contract_path: Path,
    contract: Mapping[str, Any],
    expected_feed_fingerprint: str,
    selected_action_indices: Sequence[int],
    physical_scales: Sequence[float],
) -> dict[str, Any]:
    report_path, report = _load_json(path, "standalone CPU audit report")
    if report.get("schema_version") != CPU_AUDIT_SCHEMA:
        raise ValueError("standalone CPU audit schema is incompatible")
    recorded_candidate = _resolve_recorded_path(
        report.get("candidate_path"),
        "standalone CPU audit candidate",
    )
    if recorded_candidate != Path(str(candidate_binding["path"])):
        raise ValueError("standalone CPU audit used the wrong CEM candidate")
    if report.get("candidate_file_sha256") != candidate_binding.get("sha256"):
        raise ValueError("standalone CPU audit candidate fingerprint mismatch")
    if report.get("candidate_parameter_sha256") != candidate_binding.get(
        "parameter_f32_sha256"
    ):
        raise ValueError("standalone CPU audit parameter fingerprint mismatch")
    if report.get("candidate_changed_during_audit") is not False:
        raise ValueError("CEM candidate changed during the standalone CPU audit")
    recorded_contract = _resolve_recorded_path(
        report.get("contract_path"),
        "standalone CPU audit contract",
    )
    if recorded_contract != contract_path:
        raise ValueError("standalone CPU audit used the wrong CEM contract")
    if report.get("contract_sha256") != contract.get("contract_sha256"):
        raise ValueError("standalone CPU audit contract fingerprint mismatch")
    if (
        report.get("source_feed_fingerprint") != expected_feed_fingerprint
        or report.get("audited_feed_fingerprint") != expected_feed_fingerprint
        or report.get("alternate_feed_for_unqualified_seed") is not False
    ):
        raise ValueError("standalone CPU audit used the wrong target feed")
    contract_phase = _require_finite(
        contract.get("swing_phase_advance_s"),
        "CEM swing phase advance",
    )
    source_phase = _require_finite(
        report.get("source_swing_phase_advance_s"),
        "CPU audit source swing phase advance",
    )
    audited_phase = _require_finite(
        report.get("audited_swing_phase_advance_s"),
        "CPU audit swing phase advance",
    )
    if (
        not math.isclose(source_phase, contract_phase, rel_tol=0.0, abs_tol=1e-6)
        or not math.isclose(audited_phase, contract_phase, rel_tol=0.0, abs_tol=1e-6)
        or report.get("alternate_timing_for_unqualified_seed") is not False
    ):
        raise ValueError("standalone CPU audit used the wrong control timing")
    if (
        report.get("deployment_quality_passed") is not True
        or report.get("search_margin_quality_passed") is not True
        or report.get("event_step") is None
        or report.get("hit_step") is None
        or report.get("fall_step") is not None
        or report.get("high_region_contact") is not True
    ):
        raise ValueError("standalone CPU audit did not pass the physical return gate")
    deployment = _require_mapping(
        report.get("deployment_gate"),
        "standalone CPU deployment gate",
    )
    outgoing_z = _require_finite(report.get("outgoing_z_m_s"), "CPU outgoing z")
    outgoing_forward = _require_finite(
        report.get("outgoing_forward_m_s"),
        "CPU outgoing forward velocity",
    )
    if (
        outgoing_z < _require_finite(deployment.get("min_outgoing_z_m_s"), "CPU z gate")
        or outgoing_forward
        < _require_finite(deployment.get("min_forward_m_s"), "CPU forward gate")
    ):
        raise ValueError("standalone CPU audit outgoing velocity is below its gate")
    trace_path = _resolve_recorded_path(
        report.get("trace_path"),
        "standalone CPU audit trace",
    )
    if report.get("trace_sha256") != _file_sha256(trace_path):
        raise ValueError("standalone CPU audit trace fingerprint mismatch")
    with np.load(trace_path, allow_pickle=False) as trace:
        trace_feed = str(_npz_scalar(trace, "feed_fingerprint"))
        trace_phase = float(_npz_scalar(trace, "swing_phase_advance_s"))
        trace_indices = tuple(
            int(value)
            for value in np.asarray(trace["selected_action_indices"]).tolist()
        )
        trace_scales = np.asarray(trace["physical_scales"], dtype=np.float32)
    if trace_feed != expected_feed_fingerprint:
        raise ValueError("standalone CPU trace uses the wrong target feed")
    if not math.isclose(trace_phase, contract_phase, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("standalone CPU trace uses the wrong control timing")
    if trace_indices != tuple(int(value) for value in selected_action_indices):
        raise ValueError("standalone CPU trace uses the wrong action mapping")
    if not np.allclose(
        trace_scales,
        np.asarray(physical_scales, dtype=np.float32),
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError("standalone CPU trace uses the wrong physical authority")
    return {
        **_bound_file(report_path),
        "schema_version": CPU_AUDIT_SCHEMA,
        "trace": _bound_file(trace_path),
        "candidate_parameter_f32_sha256": candidate_binding[
            "parameter_f32_sha256"
        ],
        "feed_fingerprint": expected_feed_fingerprint,
        "swing_phase_advance_s": contract_phase,
        "outgoing_z_m_s": outgoing_z,
        "outgoing_forward_m_s": outgoing_forward,
    }


def _validate_cpu_trace_dataset_identity(
    *,
    cpu_trace_path: str | Path,
    correction_dataset_path: str | Path,
) -> str:
    """Prove that the audited candidate produced the sealed BC trajectory."""

    fields = (
        "observation_normalized",
        "correction_raw",
        "correction_window",
        "time_to_intercept_s",
        "hit_event",
        "event_rebound",
        "shuttle_velocity",
        "stringbed_position",
        "right_arm_body_position_xyz_m",
        "body_fall",
        "selected_action_indices",
        "physical_scales",
        "feed_fingerprint",
        "swing_phase_advance_s",
        "outgoing_velocity_semantics",
        "event_rebound_contact_semantics",
    )
    digest = hashlib.sha256()
    with (
        np.load(Path(cpu_trace_path), allow_pickle=False) as audited,
        np.load(Path(correction_dataset_path), allow_pickle=False) as correction,
    ):
        for name in fields:
            if name not in audited.files or name not in correction.files:
                raise ValueError(
                    f"CPU-audit/correction trajectory identity is missing {name}"
                )
            audited_value = np.asarray(audited[name])
            correction_value = np.asarray(correction[name])
            if not np.array_equal(audited_value, correction_value):
                raise ValueError(
                    "standalone CPU audit and correction dataset differ at "
                    f"{name}"
                )
            digest.update(name.encode("utf-8"))
            digest.update(audited_value.dtype.str.encode("ascii"))
            digest.update(repr(audited_value.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(audited_value).tobytes(order="C"))
    return digest.hexdigest()


def build_successful_correction_dataset_manifest(
    *,
    action: str,
    expected_stage3_spec: str | Path,
    expected_feed_fingerprint: str,
    expected_control_hash: str,
    expected_latent_checkpoint_fingerprint: str | None,
    source_cem_report: str | Path,
    candidate: str | Path,
    cpu_audit_report: str | Path,
    cross_backend_seal_report: str | Path,
    correction_dataset: str | Path,
) -> dict[str, Any]:
    """Build the immutable successful-correction half of the release chain."""

    action_spec = resolve_action(action)
    if not action_spec.stage3_applicable:
        raise ValueError(f"action {action_spec.slug!r} has no Stage-3 hitting endpoint")
    registered_spec_path = _require_registered_stage3_v2_spec(
        action_spec=action_spec,
        expected_spec=expected_stage3_spec,
    )
    feed_fingerprint = _require_sha256(
        expected_feed_fingerprint,
        "expected target-feed fingerprint",
    )
    control_hash = _require_sha256(
        expected_control_hash,
        "expected control hash",
    )

    source_path, source = _load_json(source_cem_report, "source CEM report")
    if source.get("schema_version") != NATIVE_CEM_REPORT_SCHEMA:
        raise ValueError("source CEM report schema is incompatible")
    if (
        source.get("passed") is not True
        or source.get("mjx_teacher_passed") is not True
        or source.get("cpu_replay_passed") is not True
        or source.get("cpu_replay_event_equivalent") is not True
    ):
        raise ValueError("source CEM report did not pass all reachability gates")
    contract = _require_mapping(source.get("contract"), "source CEM contract")
    contract_sha256 = _validate_contract_hash(contract)
    if contract.get("feed_fingerprint") != feed_fingerprint:
        raise ValueError("source CEM report uses the wrong target feed")
    spec_identity = _validate_spec_and_scene(
        contract=contract,
        expected_spec=registered_spec_path,
    )

    candidate_path = Path(candidate).expanduser().resolve(strict=True)
    if candidate_path.parent != source_path.parent:
        raise ValueError("CEM candidate and final source report are not siblings")
    contract_path = candidate_path.parent / "cem_contract.json"
    contract_file_path, contract_file = _load_json(contract_path, "CEM contract file")
    if contract_file != contract:
        raise ValueError("source CEM report differs from its persisted contract")
    candidate_payload, candidate_binding = _validate_candidate(
        path=candidate_path,
        contract=contract,
        source_report=source,
    )

    checkpoint = _resolve_checkpoint(contract.get("source_checkpoint", ""))
    if checkpoint["payload_sha256"] != contract.get("source_checkpoint_sha256"):
        raise ValueError("CEM contract uses the wrong source checkpoint")
    checkpoint_metadata = _require_mapping(
        checkpoint["metadata"],
        "source checkpoint metadata",
    )
    control_manifest = _require_mapping(
        checkpoint_metadata.get("control_manifest"),
        "source checkpoint control manifest",
    )
    if (
        checkpoint_metadata.get("control_hash") != control_hash
        or control_manifest.get("control_hash") != control_hash
    ):
        raise ValueError("CEM source checkpoint uses the wrong control identity")
    latent_identity = _latent_identity(
        control_manifest,
        expected_latent_checkpoint_fingerprint,
    )
    bounded_residual_treatment = _bounded_residual_treatment(control_manifest)
    policy_contract = _require_mapping(
        checkpoint_metadata.get("policy_update_contract"),
        "source checkpoint policy-update contract",
    )
    _validate_policy_contract_hash(policy_contract)
    if policy_contract.get("contract_sha256") != contract.get(
        "policy_update_contract_sha256"
    ):
        raise ValueError("CEM search policy contract differs from its source checkpoint")
    selected_indices = tuple(
        int(value) for value in policy_contract.get("trainable_action_indices", ())
    )
    physical_scales = tuple(
        float(value) for value in policy_contract.get("correction_physical_scales", ())
    )
    if not selected_indices or len(selected_indices) != len(physical_scales):
        raise ValueError("source checkpoint has no complete selected-correction mapping")
    if list(contract.get("physical_scales", ())) != list(physical_scales):
        raise ValueError("CEM physical authority differs from the source checkpoint")

    cpu_binding = _validate_standalone_cpu_audit(
        path=cpu_audit_report,
        candidate_binding=candidate_binding,
        contract_path=contract_file_path,
        contract=contract,
        expected_feed_fingerprint=feed_fingerprint,
        selected_action_indices=selected_indices,
        physical_scales=physical_scales,
    )

    seal_path, seal = _load_json(
        cross_backend_seal_report,
        "cross-backend teacher seal",
    )
    if (
        seal.get("schema_version") != CROSS_BACKEND_SCHEMA
        or seal.get("passed") is not True
        or seal.get("mjx_teacher_passed") is not True
        or seal.get("cpu_replay_passed") is not True
        or seal.get("cpu_replay_event_equivalent") is not True
    ):
        raise ValueError("cross-backend teacher seal did not pass")
    if seal.get("contract") != contract:
        raise ValueError("cross-backend teacher seal uses the wrong CEM contract")
    cross_evidence = _require_mapping(
        seal.get("cross_backend_evidence"),
        "cross-backend evidence",
    )
    recorded_source = _resolve_recorded_path(
        cross_evidence.get("source_cem_report_path"),
        "cross-backend source CEM report",
    )
    if recorded_source != source_path:
        raise ValueError("cross-backend seal uses the wrong source CEM report")
    if cross_evidence.get("source_cem_report_sha256") != _file_sha256(source_path):
        raise ValueError("cross-backend source CEM report fingerprint mismatch")
    if (
        cross_evidence.get("training_backend_quality_verified") is not True
        or cross_evidence.get("feed_fingerprint") != feed_fingerprint
    ):
        raise ValueError("cross-backend seal uses the wrong target feed or backend gate")

    dataset_path = Path(correction_dataset).expanduser().resolve(strict=True)
    if dataset_path.parent != seal_path.parent:
        raise ValueError("correction dataset is not colocated with its cross-backend seal")
    trace_binding = _require_mapping(
        seal.get("teacher_trace"),
        "cross-backend teacher trajectory binding",
    )
    if (
        _resolve_recorded_path(
            trace_binding.get("trace_path"),
            "cross-backend teacher trajectory",
        )
        != dataset_path
        or trace_binding.get("trace_sha256") != _file_sha256(dataset_path)
    ):
        raise ValueError("cross-backend seal is detached from the correction dataset")
    if seal_path != dataset_path.parent / "cem_report.json":
        raise ValueError("cross-backend correction dataset requires its canonical sibling cem_report.json")
    trace_identity_sha256 = _validate_cpu_trace_dataset_identity(
        cpu_trace_path=_require_mapping(
            cpu_binding.get("trace"),
            "standalone CPU trace binding",
        )["path"],
        correction_dataset_path=dataset_path,
    )

    # Reuse the production trainer's authoritative teacher loader.  This
    # independently rechecks replica quality, CPU quality, trajectory fields,
    # action mapping, physical scales, source checkpoint, and timing semantics.
    from environment.overall_environment.src.train_incoming_hit_mjx import (
        load_quality_teacher_dataset,
    )

    teacher = load_quality_teacher_dataset(
        dataset_path,
        selected_action_indices=selected_indices,
        correction_physical_scales=physical_scales,
        source_checkpoint_sha256=checkpoint["payload_sha256"],
    )
    if (
        teacher.binding.get("schema_version") != QUALITY_TEACHER_BINDING_SCHEMA
        or teacher.binding.get("training_backend_quality_verified") is not True
        or teacher.binding.get("feed_fingerprint") != feed_fingerprint
    ):
        raise ValueError("correction dataset is not a robust quality teacher")

    target_identity = {
        "action": action_spec.slug,
        "action_slug": action_spec.slug,
        "dataset_action_id": action_spec.action_id,
        "stage3_task_action": spec_identity["stage3_task_action"],
        "stage3_task_profile": spec_identity["stage3_task_profile"],
        "stage3_experiment_id": spec_identity["stage3_experiment_id"],
        "single_feed_fingerprint": feed_fingerprint,
    }
    control_identity = {
        "control_hash": control_hash,
        "policy_update_contract_sha256": policy_contract["contract_sha256"],
        "selected_action_indices": list(selected_indices),
        "selected_actuator_names": list(
            policy_contract.get("trainable_actuator_names", ())
        ),
        "correction_physical_scales": list(physical_scales),
        "bounded_residual_treatment": bounded_residual_treatment,
        "swing_phase_advance_s": _require_finite(
            contract.get("swing_phase_advance_s"),
            "CEM swing phase advance",
        ),
    }
    unsigned: dict[str, Any] = {
        "schema_version": CORRECTION_DATASET_SCHEMA,
        "passed": True,
        "authorized_next_step": "short_bc_only",
        "target_identity": target_identity,
        "spec_identity": spec_identity,
        "control_identity": control_identity,
        "latent_identity": latent_identity,
        "source_checkpoint": {
            key: checkpoint[key]
            for key in (
                "payload_path",
                "payload_sha256",
                "metadata_path",
                "metadata_sha256",
            )
        },
        "cem": {
            "report": _bound_file(source_path),
            "contract": {
                **_bound_file(contract_file_path),
                "contract_sha256": contract_sha256,
            },
            "candidate": candidate_binding,
            "candidate_iteration": int(candidate_payload.get("iteration", -1)),
        },
        "cpu_audit": {
            **cpu_binding,
            "correction_dataset_identity_sha256": trace_identity_sha256,
        },
        "cross_backend_seal": {
            **_bound_file(seal_path),
            "schema_version": CROSS_BACKEND_SCHEMA,
            "training_backend": cross_evidence.get("training_backend"),
            "verification_source": seal.get("verification_source"),
        },
        "correction_dataset": {
            **_bound_file(dataset_path),
            "teacher_binding": teacher.binding,
        },
        "required_production_launchers": {
            "cem": "scripts/run_fullbody_training.sh --incoming-hit-cem",
            "short_bc": (
                "scripts/run_fullbody_training.sh --incoming-hit --stage train-gpu "
                "--total-env-steps 0 --curriculum-max-stage C3_static_velocity"
            ),
        },
    }
    return {
        **unsigned,
        "manifest_binding_sha256": _canonical_sha256(unsigned),
    }


def validate_successful_correction_dataset_manifest(
    path: str | Path,
) -> dict[str, Any]:
    manifest_path, manifest = _load_json(
        path,
        "successful correction dataset manifest",
    )
    if (
        manifest.get("schema_version") != CORRECTION_DATASET_SCHEMA
        or manifest.get("passed") is not True
        or manifest.get("authorized_next_step") != "short_bc_only"
    ):
        raise ValueError("successful correction dataset manifest is incompatible")
    _verify_self_hash(
        manifest,
        hash_key="manifest_binding_sha256",
        label="successful correction dataset manifest",
    )
    target = _require_mapping(manifest.get("target_identity"), "target identity")
    spec = _require_mapping(manifest.get("spec_identity"), "spec identity")
    control = _require_mapping(manifest.get("control_identity"), "control identity")
    latent = _require_mapping(manifest.get("latent_identity"), "latent identity")
    cem = _require_mapping(manifest.get("cem"), "CEM binding")
    cross = _require_mapping(manifest.get("cross_backend_seal"), "cross-backend binding")
    dataset = _require_mapping(manifest.get("correction_dataset"), "correction dataset binding")
    reconstructed = build_successful_correction_dataset_manifest(
        action=str(target.get("action", "")),
        expected_stage3_spec=str(spec.get("spec_path", "")),
        expected_feed_fingerprint=str(target.get("single_feed_fingerprint", "")),
        expected_control_hash=str(control.get("control_hash", "")),
        expected_latent_checkpoint_fingerprint=(
            None
            if latent.get("kind") == "explicit_no_latent"
            else str(latent.get("fingerprint", ""))
        ),
        source_cem_report=str(_require_mapping(cem.get("report"), "CEM report binding").get("path", "")),
        candidate=str(_require_mapping(cem.get("candidate"), "candidate binding").get("path", "")),
        cpu_audit_report=str(_require_mapping(manifest.get("cpu_audit"), "CPU audit binding").get("path", "")),
        cross_backend_seal_report=str(cross.get("path", "")),
        correction_dataset=str(dataset.get("path", "")),
    )
    if reconstructed != manifest:
        raise ValueError(
            f"successful correction dataset manifest is stale: {manifest_path}"
        )
    return manifest


def _validate_short_bc_report(
    report: Mapping[str, Any],
    *,
    teacher_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if report.get("schema_version") != SHORT_BC_REPORT_SCHEMA:
        raise ValueError("short-BC metrics schema is incompatible")
    _verify_self_hash(report, hash_key="report_sha256", label="short-BC metrics")
    if report.get("teacher_binding") != teacher_binding:
        raise ValueError("short-BC metrics use the wrong correction dataset")
    try:
        steps = int(report.get("steps"))
        batch_size = int(report.get("batch_size"))
    except (TypeError, ValueError) as exc:
        raise ValueError("short-BC metrics have invalid steps/batch size") from exc
    learning_rate = _require_finite(report.get("learning_rate"), "short-BC learning rate")
    initial = _require_finite(report.get("initial_weighted_mse"), "short-BC initial MSE")
    final = _require_finite(report.get("final_weighted_mse"), "short-BC final MSE")
    last = _require_finite(report.get("last_minibatch_mse"), "short-BC last minibatch MSE")
    improvement = _require_finite(
        report.get("improvement_fraction"),
        "short-BC improvement fraction",
    )
    expected_improvement = 0.0 if initial <= 0.0 else (initial - final) / initial
    if (
        report.get("passed") is not True
        or steps <= 0
        or batch_size <= 0
        or learning_rate <= 0.0
        or initial <= 0.0
        or final < 0.0
        or last < 0.0
        or not final < initial
        or not improvement > 0.0
        or not math.isclose(
            improvement,
            expected_improvement,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("short-BC pretraining did not produce a passing loss reduction")
    return {
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "initial_weighted_mse": initial,
        "final_weighted_mse": final,
        "improvement_fraction": improvement,
    }


def _contains_value(value: Any, expected: str) -> bool:
    if value == expected:
        return True
    if isinstance(value, Mapping):
        return any(_contains_value(item, expected) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return any(_contains_value(item, expected) for item in value)
    return False


def _validate_zero_step_train_report(report: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = _require_mapping(report, "short-BC zero-step train report")
    if (
        int(snapshot.get("requested_env_step_cap", -1)) != 0
        or int(snapshot.get("env_steps", -1)) != 0
        or int(snapshot.get("iterations", -1)) != 0
        or snapshot.get("already_at_absolute_cap") is not True
    ):
        raise ValueError("short-BC run executed PPO environment steps")
    return snapshot


def _validate_short_bc_immutable_evidence(
    *,
    correction: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate evidence carried by immutable checkpoint-version files."""

    correction_data = _require_mapping(correction, "correction dataset manifest")
    dataset_binding = _require_mapping(
        _require_mapping(
            correction_data.get("correction_dataset"),
            "correction dataset binding",
        ).get("teacher_binding"),
        "quality-teacher binding",
    )
    metrics_summary = _validate_short_bc_report(
        metrics,
        teacher_binding=dataset_binding,
    )
    if not checkpoint["versioned"]:
        raise ValueError("short-BC checkpoint must use the immutable versioned format")
    metadata = _require_mapping(checkpoint["metadata"], "short-BC checkpoint metadata")
    if (
        metadata.get("checkpoint_version") != "incoming_hit_training_v3"
        or metadata.get("checkpoint_stage") != "post_teacher_bc_pre_ppo"
        or int(metadata.get("iteration", -1)) != 0
        or int(metadata.get("env_steps", -1)) != 0
    ):
        raise ValueError("short-BC checkpoint is not the zero-PPO post-BC checkpoint")
    if metadata.get("teacher_bc_pretrain_report") != metrics:
        raise ValueError("short-BC checkpoint metadata differs from its metrics")
    config = _require_mapping(metadata.get("config"), "short-BC checkpoint config")
    if (
        config.get("policy_update_mode") != "selected_physical_correction"
        or int(config.get("total_env_steps", -1)) != 0
        or int(config.get("teacher_bc_pretrain_steps", -1)) != metrics_summary["steps"]
        or int(config.get("teacher_bc_batch_size", -1)) != metrics_summary["batch_size"]
        or not math.isclose(
            float(config.get("teacher_bc_learning_rate", math.nan)),
            metrics_summary["learning_rate"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("short-BC checkpoint config differs from its pretraining metrics")
    task_state = _require_mapping(
        metadata.get("task_curriculum_state"),
        "short-BC task-curriculum state",
    )
    if (
        task_state.get("max_stage") != "C3_static_velocity"
        or task_state.get("complete") is not False
    ):
        raise ValueError("short-BC checkpoint is not configured for the C3 static entry")
    policy_contract = _require_mapping(
        metadata.get("policy_update_contract"),
        "short-BC policy-update contract",
    )
    _validate_policy_contract_hash(policy_contract)
    correction_control = _require_mapping(
        correction_data.get("control_identity"),
        "correction control identity",
    )
    if policy_contract.get("contract_sha256") != correction_control.get(
        "policy_update_contract_sha256"
    ):
        raise ValueError("short-BC checkpoint changed the policy-update contract")
    for name in (
        "selected_action_indices",
        "selected_actuator_names",
        "correction_physical_scales",
    ):
        checkpoint_name = {
            "selected_action_indices": "trainable_action_indices",
            "selected_actuator_names": "trainable_actuator_names",
        }.get(name, name)
        if policy_contract.get(checkpoint_name) != correction_control.get(name):
            raise ValueError(f"short-BC checkpoint changed {name}")

    runtime_control = _require_mapping(
        metadata.get("control_manifest"),
        "short-BC runtime control manifest",
    )
    source_residual_treatment = _require_mapping(
        correction_control.get("bounded_residual_treatment"),
        "source bounded-residual treatment",
    )
    runtime_residual_treatment = _bounded_residual_treatment(runtime_control)
    if runtime_residual_treatment != source_residual_treatment:
        raise ValueError(
            "short-BC runtime changed the source bounded-residual treatment"
        )
    latent = _require_mapping(
        correction_data.get("latent_identity"),
        "latent identity",
    )
    _latent_identity(
        runtime_control,
        None if latent.get("kind") == "explicit_no_latent" else str(latent.get("fingerprint", "")),
    )
    runtime_feed = _require_mapping(
        metadata.get("training_feed_manifest"),
        "short-BC training feed manifest",
    )
    target = _require_mapping(
        correction_data.get("target_identity"),
        "target identity",
    )
    feed_fingerprint = str(target.get("single_feed_fingerprint", ""))
    if not _contains_value(runtime_feed, feed_fingerprint):
        raise ValueError("short-BC training feed manifest omits the reachability target feed")

    initialization = _require_mapping(
        metadata.get("actor_initialization"),
        "short-BC actor initialization",
    )
    source_checkpoint = _require_mapping(
        correction_data.get("source_checkpoint"),
        "reachability source checkpoint",
    )
    if (
        initialization.get("source_payload_sha256")
        != source_checkpoint.get("payload_sha256")
        or initialization.get("source_control_hash")
        != correction_control.get("control_hash")
        or initialization.get("runtime_control_hash")
        != runtime_control.get("control_hash")
    ):
        raise ValueError("short-BC initialized from the wrong source checkpoint")
    if initialization.get("binding_sha256") is not None:
        _verify_self_hash(
            initialization,
            hash_key="binding_sha256",
            label="short-BC actor initialization",
        )

    return {
        "metrics_summary": metrics_summary,
        "source_checkpoint": source_checkpoint,
        "runtime_control": runtime_control,
        "runtime_feed": runtime_feed,
        "policy_contract": policy_contract,
        "initialization": initialization,
    }


def _assemble_stage3_reachability_release(
    *,
    correction_path: Path,
    correction: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    evidence: Mapping[str, Any],
    metrics: Mapping[str, Any],
    metrics_source_path_at_release: str,
    metrics_source_file_sha256_at_release: str,
    train_report: Mapping[str, Any],
    train_report_source_path_at_release: str,
    train_report_source_file_sha256_at_release: str,
) -> dict[str, Any]:
    """Assemble a release without consulting mutable run-root aliases."""

    metrics_snapshot = _require_mapping(metrics, "short-BC metrics snapshot")
    train_report_snapshot = _validate_zero_step_train_report(train_report)
    metrics_summary = _require_mapping(
        evidence.get("metrics_summary"),
        "short-BC metrics summary",
    )
    source_checkpoint = _require_mapping(
        evidence.get("source_checkpoint"),
        "reachability source checkpoint",
    )
    runtime_control = _require_mapping(
        evidence.get("runtime_control"),
        "short-BC runtime control manifest",
    )
    runtime_feed = _require_mapping(
        evidence.get("runtime_feed"),
        "short-BC runtime training feed manifest",
    )
    policy_contract = _require_mapping(
        evidence.get("policy_contract"),
        "short-BC policy-update contract",
    )
    initialization = _require_mapping(
        evidence.get("initialization"),
        "short-BC actor initialization",
    )
    immutable_checkpoint_keys = (
        "payload_path",
        "payload_sha256",
        "metadata_path",
        "metadata_sha256",
        "completion_path",
        "completion_sha256",
    )

    unsigned: dict[str, Any] = {
        "schema_version": REACHABILITY_RELEASE_SCHEMA,
        "passed": True,
        "authorized_next_step": "stage3_static_single_feed_ppo",
        "scope_limitations": [
            "does_not_claim_static_ppo_success",
            "does_not_claim_multi_feed_generalization",
            "does_not_claim_heldout_feed_success",
        ],
        "target_identity": correction["target_identity"],
        "spec_identity": correction["spec_identity"],
        "latent_identity": correction["latent_identity"],
        "control_identity": correction["control_identity"],
        "correction_dataset_manifest": {
            **_bound_file(correction_path),
            "manifest_binding_sha256": correction["manifest_binding_sha256"],
        },
        "short_bc": {
            "checkpoint": {
                key: checkpoint[key] for key in immutable_checkpoint_keys
            },
            "metrics": {
                "source_path_at_release": metrics_source_path_at_release,
                "source_file_sha256_at_release": (
                    metrics_source_file_sha256_at_release
                ),
                "snapshot": metrics_snapshot,
                "snapshot_sha256": _canonical_sha256(metrics_snapshot),
                **metrics_summary,
            },
            "zero_step_train_report_snapshot": {
                "source_path_at_release": train_report_source_path_at_release,
                "source_file_sha256_at_release": (
                    train_report_source_file_sha256_at_release
                ),
                "snapshot": train_report_snapshot,
                "snapshot_sha256": _canonical_sha256(train_report_snapshot),
            },
            "source_checkpoint_payload_sha256": source_checkpoint["payload_sha256"],
            "runtime_control_manifest": runtime_control,
            "runtime_control_manifest_sha256": _canonical_sha256(runtime_control),
            "runtime_training_feed_manifest": runtime_feed,
            "runtime_training_feed_manifest_sha256": _canonical_sha256(runtime_feed),
            "policy_update_contract": policy_contract,
            "actor_initialization": initialization,
            "ppo_environment_steps": 0,
        },
        "required_production_launchers": correction[
            "required_production_launchers"
        ],
    }
    return {
        **unsigned,
        "release_binding_sha256": _canonical_sha256(unsigned),
    }


def build_stage3_reachability_release(
    *,
    correction_dataset_manifest: str | Path,
    short_bc_checkpoint: str | Path,
    short_bc_metrics: str | Path,
) -> dict[str, Any]:
    """Seal a successful correction dataset and pure short-BC checkpoint.

    The mutable latest pointer and current train report are checked only at
    this initial sealing boundary.  The returned release permanently carries
    the exact immutable checkpoint-version files and an inline zero-step
    report snapshot, so later PPO checkpoints in the same run cannot
    invalidate their own ancestor.
    """

    correction_path = Path(correction_dataset_manifest).expanduser().resolve(strict=True)
    correction = validate_successful_correction_dataset_manifest(correction_path)
    metrics_path, metrics = _load_json(short_bc_metrics, "short-BC metrics")
    checkpoint = _resolve_checkpoint(short_bc_checkpoint)
    evidence = _validate_short_bc_immutable_evidence(
        correction=correction,
        checkpoint=checkpoint,
        metrics=metrics,
    )

    payload_path = Path(checkpoint["payload_path"])
    run_root = payload_path.parent.parent.parent
    canonical_metrics_path = run_root / "teacher_bc_pretrain_report.json"
    if metrics_path != canonical_metrics_path.resolve(strict=True):
        raise ValueError("short-BC metrics are not colocated with the checkpoint run")
    train_report_path, train_report = _load_json(
        run_root / "train_report.json",
        "short-BC train report",
    )
    _validate_zero_step_train_report(train_report)
    latest = _resolve_checkpoint(run_root / "policy_latest.json")
    immutable_keys = (
        "payload_path",
        "payload_sha256",
        "metadata_path",
        "metadata_sha256",
        "completion_path",
        "completion_sha256",
    )
    if any(latest[key] != checkpoint[key] for key in immutable_keys):
        raise ValueError("short-BC checkpoint is not the run's immutable latest pointer")

    return _assemble_stage3_reachability_release(
        correction_path=correction_path,
        correction=correction,
        checkpoint=checkpoint,
        evidence=evidence,
        metrics=metrics,
        metrics_source_path_at_release=str(metrics_path),
        metrics_source_file_sha256_at_release=_file_sha256(metrics_path),
        train_report=train_report,
        train_report_source_path_at_release=str(train_report_path),
        train_report_source_file_sha256_at_release=_file_sha256(train_report_path),
    )


def validate_stage3_reachability_release(path: str | Path) -> dict[str, Any]:
    release_path, release = _load_json(path, "Stage-3 reachability release")
    if (
        release.get("schema_version") != REACHABILITY_RELEASE_SCHEMA
        or release.get("passed") is not True
        or release.get("authorized_next_step") != "stage3_static_single_feed_ppo"
    ):
        raise ValueError("Stage-3 reachability release is incompatible")
    _verify_self_hash(
        release,
        hash_key="release_binding_sha256",
        label="Stage-3 reachability release",
    )
    correction = _require_mapping(
        release.get("correction_dataset_manifest"),
        "release correction-dataset binding",
    )
    correction_path = _resolve_recorded_path(
        correction.get("path"),
        "released correction-dataset manifest",
    )
    correction_manifest = validate_successful_correction_dataset_manifest(
        correction_path
    )
    expected_correction_binding = {
        **_bound_file(correction_path),
        "manifest_binding_sha256": correction_manifest[
            "manifest_binding_sha256"
        ],
    }
    if correction != expected_correction_binding:
        raise ValueError("Stage-3 release correction-dataset binding is stale")

    short_bc = _require_mapping(release.get("short_bc"), "release short-BC binding")
    checkpoint_binding = _require_mapping(
        short_bc.get("checkpoint"),
        "short-BC checkpoint binding",
    )
    checkpoint = _resolve_checkpoint(str(checkpoint_binding.get("payload_path", "")))
    immutable_keys = (
        "payload_path",
        "payload_sha256",
        "metadata_path",
        "metadata_sha256",
        "completion_path",
        "completion_sha256",
    )
    expected_checkpoint_binding = {key: checkpoint[key] for key in immutable_keys}
    if checkpoint_binding != expected_checkpoint_binding:
        raise ValueError("Stage-3 release immutable short-BC checkpoint changed")

    metrics_binding = _require_mapping(
        short_bc.get("metrics"),
        "short-BC metrics snapshot binding",
    )
    metrics = _require_mapping(
        metrics_binding.get("snapshot"),
        "short-BC metrics snapshot",
    )
    if metrics_binding.get("snapshot_sha256") != _canonical_sha256(metrics):
        raise ValueError("short-BC metrics snapshot fingerprint mismatch")
    _require_sha256(
        metrics_binding.get("source_file_sha256_at_release"),
        "short-BC metrics source fingerprint",
    )
    run_root = Path(checkpoint["payload_path"]).parent.parent.parent
    metrics_source_path = Path(
        str(metrics_binding.get("source_path_at_release", ""))
    ).expanduser()
    if (
        not metrics_source_path.is_absolute()
        or metrics_source_path.resolve()
        != (run_root / "teacher_bc_pretrain_report.json").resolve()
    ):
        raise ValueError("short-BC metrics snapshot has the wrong source path")

    train_binding = _require_mapping(
        short_bc.get("zero_step_train_report_snapshot"),
        "short-BC zero-step train-report snapshot binding",
    )
    train_report = _require_mapping(
        train_binding.get("snapshot"),
        "short-BC zero-step train-report snapshot",
    )
    _validate_zero_step_train_report(train_report)
    if train_binding.get("snapshot_sha256") != _canonical_sha256(train_report):
        raise ValueError("short-BC zero-step train-report snapshot fingerprint mismatch")
    _require_sha256(
        train_binding.get("source_file_sha256_at_release"),
        "short-BC zero-step train-report source fingerprint",
    )
    train_source_path = Path(
        str(train_binding.get("source_path_at_release", ""))
    ).expanduser()
    if (
        not train_source_path.is_absolute()
        or train_source_path.resolve() != (run_root / "train_report.json").resolve()
    ):
        raise ValueError("short-BC zero-step train-report snapshot has the wrong source path")

    evidence = _validate_short_bc_immutable_evidence(
        correction=correction_manifest,
        checkpoint=checkpoint,
        metrics=metrics,
    )
    reconstructed = _assemble_stage3_reachability_release(
        correction_path=correction_path,
        correction=correction_manifest,
        checkpoint=checkpoint,
        evidence=evidence,
        metrics=metrics,
        metrics_source_path_at_release=str(metrics_source_path),
        metrics_source_file_sha256_at_release=str(
            metrics_binding.get("source_file_sha256_at_release")
        ),
        train_report=train_report,
        train_report_source_path_at_release=str(train_source_path),
        train_report_source_file_sha256_at_release=str(
            train_binding.get("source_file_sha256_at_release")
        ),
    )
    if reconstructed != release:
        raise ValueError(f"Stage-3 reachability release is stale: {release_path}")
    return release


def validate_static_ppo_entry(
    *,
    release_path: str | Path,
    start_checkpoint: str | Path,
    teacher_dataset: str | Path,
    runtime_run_dir: str | Path,
    runtime_control_manifest: Mapping[str, Any],
    runtime_training_feed_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the only allowed transition from reachability to static PPO."""

    path = Path(release_path).expanduser().resolve(strict=True)
    release = validate_stage3_reachability_release(path)
    short_bc = _require_mapping(release.get("short_bc"), "release short-BC binding")
    released_checkpoint = _require_mapping(
        short_bc.get("checkpoint"),
        "released short-BC checkpoint",
    )
    initialization = _resolve_checkpoint(start_checkpoint)
    if (
        initialization["payload_path"] != released_checkpoint.get("payload_path")
        or initialization["payload_sha256"] != released_checkpoint.get("payload_sha256")
        or initialization["metadata_sha256"] != released_checkpoint.get("metadata_sha256")
    ):
        raise ValueError("static PPO resumed from the wrong short-BC checkpoint")
    correction_manifest_binding = _require_mapping(
        release.get("correction_dataset_manifest"),
        "released correction-dataset manifest",
    )
    correction_manifest = validate_successful_correction_dataset_manifest(
        str(correction_manifest_binding.get("path", ""))
    )
    released_dataset = _require_mapping(
        correction_manifest.get("correction_dataset"),
        "released correction dataset",
    )
    dataset_path = Path(teacher_dataset).expanduser().resolve(strict=True)
    if (
        str(dataset_path) != released_dataset.get("path")
        or _file_sha256(dataset_path) != released_dataset.get("sha256")
    ):
        raise ValueError("static PPO uses the wrong sealed correction dataset")
    payload_path = Path(initialization["payload_path"])
    released_run_root = payload_path.parent.parent.parent.resolve(strict=True)
    run_root = Path(runtime_run_dir).expanduser().resolve(strict=True)
    if run_root != released_run_root:
        raise ValueError("static PPO must continue in the short-BC run root")
    runtime_control = _require_mapping(runtime_control_manifest, "static PPO control manifest")
    runtime_feed = _require_mapping(
        runtime_training_feed_manifest,
        "static PPO training feed manifest",
    )
    if (
        runtime_control != short_bc.get("runtime_control_manifest")
        or _canonical_sha256(runtime_control)
        != short_bc.get("runtime_control_manifest_sha256")
    ):
        raise ValueError("static PPO control identity differs from short BC")
    if (
        runtime_feed != short_bc.get("runtime_training_feed_manifest")
        or _canonical_sha256(runtime_feed)
        != short_bc.get("runtime_training_feed_manifest_sha256")
    ):
        raise ValueError("static PPO feed identity differs from short BC")
    unsigned = {
        "schema_version": STATIC_PPO_ENTRY_SCHEMA,
        "verified": True,
        "release_path": str(path),
        "release_file_sha256": _file_sha256(path),
        "release_binding_sha256": release["release_binding_sha256"],
        "static_start_mode": "resume_exact_zero_ppo_post_bc_checkpoint",
        "static_start_checkpoint_payload_path": initialization["payload_path"],
        "static_start_checkpoint_payload_sha256": initialization["payload_sha256"],
        "static_start_checkpoint_metadata_sha256": initialization["metadata_sha256"],
        "sealed_correction_dataset_path": str(dataset_path),
        "sealed_correction_dataset_sha256": _file_sha256(dataset_path),
        "short_bc_and_static_run_root": str(run_root),
        "runtime_control_manifest_sha256": _canonical_sha256(runtime_control),
        "runtime_training_feed_manifest_sha256": _canonical_sha256(runtime_feed),
        "authorized_stage": "stage3_static_single_feed_ppo",
    }
    return {**unsigned, "binding_sha256": _canonical_sha256(unsigned)}


def validate_post_static_ppo_continuation(
    *,
    release_path: str | Path,
    static_checkpoint: str | Path,
    teacher_dataset: str | Path,
    runtime_run_dir: str | Path,
) -> dict[str, Any]:
    """Recover a release entry only from a completed, lineage-bound C3 run."""

    path = Path(release_path).expanduser().resolve(strict=True)
    release = validate_stage3_reachability_release(path)
    short_bc = _require_mapping(release.get("short_bc"), "release short-BC binding")
    released_checkpoint = _require_mapping(
        short_bc.get("checkpoint"),
        "released short-BC checkpoint",
    )
    expected_entry = validate_static_ppo_entry(
        release_path=path,
        start_checkpoint=str(
            released_checkpoint.get("pointer_path")
            or released_checkpoint.get("payload_path")
            or ""
        ),
        teacher_dataset=str(
            _require_mapping(
                validate_successful_correction_dataset_manifest(
                    _require_mapping(
                        release.get("correction_dataset_manifest"),
                        "released correction-dataset manifest",
                    )["path"]
                ).get("correction_dataset"),
                "released correction dataset",
            )["path"]
        ),
        runtime_run_dir=str(
            Path(str(released_checkpoint.get("payload_path", "")))
            .parent.parent.parent
        ),
        runtime_control_manifest=_require_mapping(
            short_bc.get("runtime_control_manifest"),
            "released short-BC control manifest",
        ),
        runtime_training_feed_manifest=_require_mapping(
            short_bc.get("runtime_training_feed_manifest"),
            "released short-BC feed manifest",
        ),
    )
    checkpoint = _resolve_checkpoint(static_checkpoint)
    if not checkpoint["versioned"]:
        raise ValueError("post-static continuation requires a versioned C3 checkpoint")
    metadata = _require_mapping(
        checkpoint.get("metadata"),
        "static PPO checkpoint metadata",
    )
    task_state = _require_mapping(
        metadata.get("task_curriculum_state"),
        "static PPO task-curriculum state",
    )
    if (
        metadata.get("checkpoint_stage") != "ppo_iteration_boundary"
        or int(metadata.get("env_steps", 0)) <= 0
        or task_state.get("max_stage") != "C3_static_velocity"
        or task_state.get("complete") is not True
    ):
        raise ValueError(
            "post-static continuation requires a completed C3_static_velocity PPO checkpoint"
        )
    prerequisites = _require_mapping(
        metadata.get("training_prerequisite_binding"),
        "static PPO prerequisite binding",
    )
    _verify_self_hash(
        prerequisites,
        hash_key="binding_sha256",
        label="static PPO prerequisite binding",
    )
    recorded_entry = _require_mapping(
        prerequisites.get("stage3_reachability_release"),
        "static PPO reachability lineage",
    )
    if recorded_entry != expected_entry:
        raise ValueError("static PPO checkpoint has the wrong reachability-release lineage")
    dataset_path = Path(teacher_dataset).expanduser().resolve(strict=True)
    if (
        str(dataset_path) != recorded_entry.get("sealed_correction_dataset_path")
        or _file_sha256(dataset_path)
        != recorded_entry.get("sealed_correction_dataset_sha256")
    ):
        raise ValueError("post-static PPO uses the wrong sealed correction dataset")
    run_root = Path(runtime_run_dir).expanduser().resolve(strict=True)
    if str(run_root) != recorded_entry.get("short_bc_and_static_run_root"):
        raise ValueError("post-static PPO must remain in the short-BC/C3 run root")
    return recorded_entry


def attach_static_ppo_entry_to_prerequisites(
    prerequisite_binding: Mapping[str, Any],
    entry_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Add the reachability entry to an existing hashed prerequisite binding."""

    existing = _require_mapping(prerequisite_binding, "Stage-3 prerequisite binding")
    if existing.get("verified") is not True:
        raise ValueError("Stage-3 prerequisite binding is not verified")
    if existing.get("binding_sha256") is not None:
        _verify_self_hash(
            existing,
            hash_key="binding_sha256",
            label="Stage-3 prerequisite binding",
        )
    entry = _require_mapping(entry_binding, "static PPO entry binding")
    if (
        entry.get("schema_version") != STATIC_PPO_ENTRY_SCHEMA
        or entry.get("verified") is not True
    ):
        raise ValueError("static PPO entry binding is incompatible")
    _verify_self_hash(entry, hash_key="binding_sha256", label="static PPO entry binding")
    unsigned = dict(existing)
    unsigned.pop("binding_sha256", None)
    unsigned["stage3_reachability_release"] = entry
    return {**unsigned, "binding_sha256": _canonical_sha256(unsigned)}


def validate_static_ppo_prerequisite_extension(
    *,
    checkpoint_binding: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
    checkpoint_payload_sha256: str,
) -> None:
    """Allow the one acyclic BC->C3 prerequisite extension, and nothing else."""

    checkpoint = _require_mapping(
        checkpoint_binding,
        "short-BC checkpoint prerequisite binding",
    )
    runtime = _require_mapping(
        runtime_binding,
        "static PPO runtime prerequisite binding",
    )
    for label, binding in (
        ("short-BC checkpoint prerequisite binding", checkpoint),
        ("static PPO runtime prerequisite binding", runtime),
    ):
        if binding.get("verified") is not True:
            raise ValueError(f"{label} is not verified")
        _verify_self_hash(binding, hash_key="binding_sha256", label=label)
    entry = _require_mapping(
        runtime.get("stage3_reachability_release"),
        "static PPO reachability entry",
    )
    release = validate_stage3_reachability_release(
        str(entry.get("release_path", ""))
    )
    short_bc = _require_mapping(release.get("short_bc"), "release short-BC binding")
    expected_entry = validate_static_ppo_entry(
        release_path=str(entry.get("release_path", "")),
        start_checkpoint=str(entry.get("static_start_checkpoint_payload_path", "")),
        teacher_dataset=str(entry.get("sealed_correction_dataset_path", "")),
        runtime_run_dir=str(entry.get("short_bc_and_static_run_root", "")),
        runtime_control_manifest=_require_mapping(
            short_bc.get("runtime_control_manifest"),
            "released short-BC control manifest",
        ),
        runtime_training_feed_manifest=_require_mapping(
            short_bc.get("runtime_training_feed_manifest"),
            "released short-BC feed manifest",
        ),
    )
    if entry != expected_entry:
        raise ValueError("static PPO runtime has an invalid reachability entry")
    if entry.get("static_start_checkpoint_payload_sha256") != _require_sha256(
        checkpoint_payload_sha256,
        "resumed short-BC checkpoint payload fingerprint",
    ):
        raise ValueError("static PPO did not resume the released short-BC checkpoint")
    stripped = dict(runtime)
    stripped.pop("binding_sha256", None)
    stripped.pop("stage3_reachability_release", None)
    stripped["binding_sha256"] = _canonical_sha256(stripped)
    if stripped != checkpoint:
        raise ValueError(
            "static PPO changed prerequisites beyond adding its reachability release"
        )


def _validate_extra_launch_args(
    extra_args: Sequence[str],
    *,
    reserved: frozenset[str],
) -> tuple[str, ...]:
    values = tuple(str(value) for value in extra_args)
    for value in values:
        option = value.split("=", 1)[0]
        if option in reserved:
            raise ValueError(
                f"extra launch arguments cannot override sealed option {option}"
            )
    return values


def canonical_cem_launch_command(
    *,
    spec: str | Path,
    checkpoint: str | Path,
    out_dir: str | Path,
    extra_args: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return the only production command prefix allowed for CEM search."""

    extra = _validate_extra_launch_args(
        extra_args,
        reserved=frozenset({"--spec", "--checkpoint", "--out-dir"}),
    )
    return (
        "scripts/run_fullbody_training.sh",
        "--incoming-hit-cem",
        "--spec",
        str(spec),
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        str(out_dir),
        *extra,
    )


def canonical_short_bc_launch_command(
    *,
    spec: str | Path,
    source_checkpoint: str | Path,
    correction_dataset: str | Path,
    out_dir: str | Path,
    extra_args: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return a canonical pure-BC launch (zero PPO environment steps)."""

    extra = _validate_extra_launch_args(
        extra_args,
        reserved=frozenset(
            {
                "--spec",
                "--stage",
                "--initialize-policy-from",
                "--resume-from",
                "--teacher-dataset",
                "--exploration-prior-dataset",
                "--total-env-steps",
                "--curriculum-max-stage",
                "--out-dir",
            }
        ),
    )
    return (
        "scripts/run_fullbody_training.sh",
        "--incoming-hit",
        "--spec",
        str(spec),
        "--stage",
        "train-gpu",
        "--initialize-policy-from",
        str(source_checkpoint),
        "--teacher-dataset",
        str(correction_dataset),
        "--total-env-steps",
        "0",
        "--curriculum-max-stage",
        "C3_static_velocity",
        "--out-dir",
        str(out_dir),
        *extra,
    )


def _latent_cli_value(args: argparse.Namespace) -> str | None:
    return None if args.expect_no_latent else str(args.expected_latent_fingerprint)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser(
        "build-dataset-manifest",
        help="seal CEM/CPU/cross-backend evidence and the successful correction dataset",
    )
    dataset.add_argument("--action", choices=action_choices(), required=True)
    dataset.add_argument("--expected-stage3-spec", required=True)
    dataset.add_argument("--expected-feed-fingerprint", required=True)
    dataset.add_argument("--expected-control-hash", required=True)
    latent = dataset.add_mutually_exclusive_group(required=True)
    latent.add_argument("--expected-latent-fingerprint")
    latent.add_argument("--expect-no-latent", action="store_true")
    dataset.add_argument("--source-cem-report", required=True)
    dataset.add_argument("--candidate", required=True)
    dataset.add_argument("--cpu-audit-report", required=True)
    dataset.add_argument("--cross-backend-seal-report", required=True)
    dataset.add_argument("--correction-dataset", required=True)
    dataset.add_argument("--output", required=True)

    release = subparsers.add_parser(
        "build-release",
        help="seal the successful correction manifest and zero-PPO short-BC checkpoint",
    )
    release.add_argument("--correction-dataset-manifest", required=True)
    release.add_argument("--short-bc-checkpoint", required=True)
    release.add_argument("--short-bc-metrics", required=True)
    release.add_argument("--output", required=True)

    validate = subparsers.add_parser("validate", help="revalidate a complete release")
    validate.add_argument("--release", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "build-dataset-manifest":
        payload = build_successful_correction_dataset_manifest(
            action=args.action,
            expected_stage3_spec=args.expected_stage3_spec,
            expected_feed_fingerprint=args.expected_feed_fingerprint,
            expected_control_hash=args.expected_control_hash,
            expected_latent_checkpoint_fingerprint=_latent_cli_value(args),
            source_cem_report=args.source_cem_report,
            candidate=args.candidate,
            cpu_audit_report=args.cpu_audit_report,
            cross_backend_seal_report=args.cross_backend_seal_report,
            correction_dataset=args.correction_dataset,
        )
        _atomic_write_json(args.output, payload)
    elif args.command == "build-release":
        payload = build_stage3_reachability_release(
            correction_dataset_manifest=args.correction_dataset_manifest,
            short_bc_checkpoint=args.short_bc_checkpoint,
            short_bc_metrics=args.short_bc_metrics,
        )
        _atomic_write_json(args.output, payload)
    else:
        payload = validate_stage3_reachability_release(args.release)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
