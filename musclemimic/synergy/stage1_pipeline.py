"""Transactional Stage-1 early-synergy preparation and preflight.

The pipeline deliberately separates *artifact completeness* from *permission to
train*.  An object may be atomically published at ``basis_ready`` after a valid
primitive fit, but formal S/SR bindings are emitted only after an independently
provenanced target-control proxy passes coverage and the frozen runtime action
interface can be rebuilt offline.

Artifacts are built in an immutable, never-moved ``.objects/<input hash>``
directory.  This matters because the regional-composite basis contract stores
absolute paths to its component artifacts.  Publication consists of atomic
``release.json``, ``READY.json`` and release-pointer writes; incomplete object
directories are never referenced by a release pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.synergy.action_interface import (
    build_early_synergy_action_interface,
    load_coefficient_statistics,
)
from musclemimic.synergy.basis_artifact import load_synergy_basis
from musclemimic.synergy.fit import (
    EXCITATION_SIGNAL_KIND,
    BasisNotEligibleForEarlyControl,
    SynergyFitConfig,
    fit_synergy_dataset,
    load_synergy_split,
)
from musclemimic.synergy.frozen_decoder import load_frozen_body_decoder
from musclemimic.synergy.oracle_coverage import (
    StaticProxyCoverageThresholds,
    evaluate_static_proxy_coverage,
    load_static_proxy_phase_schema,
    write_static_proxy_coverage_gate,
)
from musclemimic.synergy.primitive_manifest import (
    load_primitive_source_manifest,
    save_primitive_source_manifest_from_splits,
)
from musclemimic.synergy.residual_fit import (
    StructuredResidualFitConfig,
    fit_structured_residual_basis,
)
from musclemimic.synergy.schema import ctrlrange_schema_hash
from musclemimic.synergy.semantic_contracts import (
    primitive_semantic_contracts,
    validate_primitive_semantic_contracts,
)

PIPELINE_PLAN_SCHEMA_VERSION = "stage1_synergy_pipeline_plan_v1"
PIPELINE_RELEASE_SCHEMA_VERSION = "stage1_synergy_pipeline_release_v1"
PIPELINE_READY_SCHEMA_VERSION = "stage1_synergy_pipeline_ready_v1"
PIPELINE_POINTER_SCHEMA_VERSION = "stage1_synergy_release_pointer_v1"
PIPELINE_BINDINGS_SCHEMA_VERSION = "stage1_synergy_pipeline_bindings_v1"
PIPELINE_PREFLIGHT_SCHEMA_VERSION = "stage1_synergy_pipeline_preflight_v1"

READINESS_ORDER = {
    "source_validated": 0,
    "basis_ready": 1,
    "coverage_ready": 2,
    "training_ready_s": 3,
    "training_ready_sr": 4,
}
TRAINING_READINESS = frozenset({"training_ready_s", "training_ready_sr"})
CANONICAL_HYDRA_OVERRIDES = ("config_status.allow_nonproduction_runtime=true",)
BOOTSTRAP_EVIDENCE_LIMITATIONS = ("no_independent_chinajump_target_control_coverage",)

DEFAULT_FORMAL_CONFIG = "config_specific_task/stage1_body/conf_fullbody_chinajump_early_synergy"
DEFAULT_RESIDUAL_CONFIG = "config_specific_task/stage1_body/conf_fullbody_chinajump_early_synergy_residual"
DEFAULT_BOOTSTRAP_CONFIG = "config_specific_task/stage1_body/conf_fullbody_chinajump_early_synergy_bootstrap"
DEFAULT_PHASE_SCHEMA = "fullbody/config_specific_task/stage1_body/chinajump_coverage_phase_schema_v1.json"
DEFAULT_GROUPING = "experiments/synergy/forehand_clear_myofullbody_354_regions_v1.json"
DEFAULT_OUTPUT_ROOT = "artifacts/stage1_synergy/chinajump_v1"
DEFAULT_ENV_PREFIX = "MUSCLEMIMIC_CHINAJUMP"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SAFE_SLUG_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
_FORMAL_PROXY_FORBIDDEN_SOURCE_TOKENS = (
    "primitive",
    "early_synergy",
    "fixed_synergy",
    "synergy_policy",
)


@dataclass(frozen=True)
class PipelineRequest:
    """Canonical user request used to derive an immutable input identity."""

    train: str | None
    val: str | None
    primitive_catalog: str | None
    grouping_json: str
    coverage_proxy_artifact: str | None
    phase_schema: str
    residual_mask: str | None
    output_root: str
    target_skill_id: str
    env_prefix: str
    readiness_mode: str
    with_residual: bool
    formal_config_name: str
    residual_config_name: str
    bootstrap_config_name: str
    ranks: tuple[int, ...]
    seeds: tuple[int, ...]
    normalization: str
    near_zero_threshold: float
    phase_weights_json: str | None
    fit_mode: str = "both"
    region_ranks: Mapping[str, tuple[int, ...]] | None = None
    total_rank_budget: int | None = None
    require_dynamic_coverage: bool = False
    max_mean_dynamic_gap: float = 0.15
    max_key_phase_dynamic_gap: float = 0.25
    max_basis_condition_number: float = 1.0e6
    min_effective_rank_fraction: float = 1.0
    expected_environment_fingerprint: str | None = None
    expected_rollout_manifest_fingerprint: str | None = None
    dynamic_coverage_reports: (
        Mapping[
            str,
            Mapping[str, Mapping[int | str, Mapping[str, Any]]],
        ]
        | None
    ) = None


@dataclass(frozen=True)
class PrimitiveSourceView:
    train_source: Path
    validation_source: Path
    source_checkpoints_path: Path | None
    regional_grouping_path: Path
    metadata: dict[str, Any]
    source_identity: dict[str, Any]


@dataclass(frozen=True)
class CoverageProxyView:
    npz_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    fingerprint: str
    source_kind: str
    producer_binding: dict[str, Any]
    required_phase_ids: tuple[int, ...]
    min_phase_samples: int
    per_phase_sample_counts: dict[int, int]


class PipelineInputError(ValueError):
    """Raised when a caller-supplied artifact or request is invalid."""


class IncompletePipelineObjectError(RuntimeError):
    """Raised instead of reusing an object left by a failed/interrupted apply."""


def plan_stage1_pipeline(request: PipelineRequest) -> dict[str, Any]:
    """Read and validate available inputs without creating any filesystem entry."""

    req = _validate_request(request)
    missing: list[str] = []
    warnings: list[str] = []
    source_identity: dict[str, Any] | None = None
    runtime_contract: dict[str, Any] | None = None

    if req.primitive_catalog is not None:
        catalog_path = Path(req.primitive_catalog).resolve()
        if not catalog_path.is_file():
            missing.append(f"primitive_catalog:{catalog_path}")
        else:
            catalog = _load_primitive_catalog(catalog_path, require_build_ready=False)
            try:
                _validate_build_ready_catalog(catalog)
                build_ready = True
                build_error = None
            except (FileNotFoundError, ValueError) as exc:
                build_ready = False
                build_error = str(exc)
            source_identity = {
                "kind": "primitive_catalog",
                "path": str(catalog_path),
                "fingerprint": _require_sha256(
                    getattr(catalog, "fingerprint", None),
                    "primitive catalog fingerprint",
                ),
                "target_skill_id": str(getattr(catalog, "target_skill_id", "")),
                "expected_action_dim": int(getattr(catalog, "expected_action_dim", 0)),
                "build_ready": build_ready,
                "build_input_identity": _catalog_build_input_identity(catalog),
            }
            if not source_identity["build_ready"]:
                missing.append(f"primitive_catalog_not_build_ready:{build_error}")
    else:
        train_path = _required_existing_path(req.train, "train source", missing)
        val_path = _required_existing_path(req.val, "validation source", missing)
        if train_path is not None and val_path is not None:
            train = load_synergy_split(train_path, split="train")
            validation = load_synergy_split(val_path, split="val")
            if train.muscle_names != validation.muscle_names:
                raise PipelineInputError("primitive train/validation ordered actuator names differ")
            source_identity = {
                "kind": "prepared_primitive_shards",
                "train_path": str(train_path),
                "validation_path": str(val_path),
                "train_content_fingerprint": train.content_fingerprint,
                "validation_content_fingerprint": validation.content_fingerprint,
            }
            runtime_contract = _runtime_contract_from_splits(train, validation)
            source_identity["primitive_semantic_contracts"] = runtime_contract["primitive_semantic_contracts"]
            source_identity["primitive_semantic_attestation"] = runtime_contract["primitive_semantic_attestation"]

    grouping_path = Path(req.grouping_json).resolve()
    if req.primitive_catalog is not None and source_identity is not None:
        catalog_grouping = source_identity["build_input_identity"].get("regional_grouping_path")
        if catalog_grouping:
            grouping_path = Path(str(catalog_grouping)).resolve()
    grouping_identity = _file_identity_or_missing(
        grouping_path,
        label="regional_grouping",
        missing=missing,
    )
    phase_schema_path = Path(req.phase_schema).resolve()
    phase_schema_identity = _file_identity_or_missing(
        phase_schema_path,
        label="coverage_phase_schema",
        missing=missing,
    )
    if phase_schema_identity is not None:
        phase_schema = load_static_proxy_phase_schema(phase_schema_path)
        phase_schema_identity["semantic_fingerprint"] = phase_schema["phase_schema_fingerprint"]
        if phase_schema["target_skill_id"] != req.target_skill_id:
            raise PipelineInputError("coverage phase schema target_skill_id differs from pipeline target")

    residual_identity: dict[str, Any] | None = None
    if req.with_residual:
        if req.residual_mask is None:
            missing.append("residual_mask")
        else:
            residual_identity = _file_identity_or_missing(
                Path(req.residual_mask).resolve(),
                label="residual_mask",
                missing=missing,
            )

    proxy_identity: dict[str, Any] | None = None
    proxy_view: CoverageProxyView | None = None
    if req.coverage_proxy_artifact is not None:
        proxy_path = Path(req.coverage_proxy_artifact).resolve()
        if not proxy_path.exists():
            missing.append(f"coverage_proxy_artifact:{proxy_path}")
        elif proxy_path.suffix == ".npz":
            raise PipelineInputError("formal coverage requires a sealed proxy artifact, never a bare NPZ")
        else:
            proxy_view = _load_coverage_proxy_artifact(
                proxy_path,
                expected_target_skill_id=req.target_skill_id,
                expected_runtime_contract=runtime_contract,
                expected_phase_schema_fingerprint=(
                    None if phase_schema_identity is None else str(phase_schema_identity["semantic_fingerprint"])
                ),
            )
            proxy_identity = {
                "path": str(proxy_view.manifest_path),
                "fingerprint": proxy_view.fingerprint,
                "npz_path": str(proxy_view.npz_path),
                "npz_sha256": _file_sha256(proxy_view.npz_path),
                "source_kind": proxy_view.source_kind,
                "required_phase_ids": list(proxy_view.required_phase_ids),
                "min_phase_samples": proxy_view.min_phase_samples,
                "per_phase_sample_counts": {
                    str(key): value for key, value in sorted(proxy_view.per_phase_sample_counts.items())
                },
            }
    elif req.readiness_mode == "formal":
        warnings.append(
            "formal target-control proxy is absent; apply can publish basis_ready "
            "but cannot emit formal training bindings"
        )

    config_name = (
        req.bootstrap_config_name
        if req.readiness_mode == "bootstrap"
        else req.residual_config_name
        if req.with_residual
        else req.formal_config_name
    )
    config_contract = _pipeline_config_contract(
        config_name,
        env_prefix=req.env_prefix,
        readiness_mode=req.readiness_mode,
    )
    if proxy_view is not None:
        _validate_formal_proxy_phase_contract(proxy_view, config_contract)
    _validate_plan_runtime_contract(
        request=req,
        config_contract=config_contract,
        source_identity=source_identity,
        runtime_contract=runtime_contract,
        phase_schema_identity=phase_schema_identity,
    )
    phase_weights_identity = (
        None
        if req.phase_weights_json is None
        else _file_identity_or_missing(
            Path(req.phase_weights_json).resolve(),
            label="phase_weights",
            missing=missing,
        )
    )
    if req.require_dynamic_coverage and req.dynamic_coverage_reports is None:
        warnings.append(
            "dynamic coverage is required but no external reports were supplied; "
            "apply will persist deterministic candidate artifacts and stop for "
            "second-stage environment-rollout evidence"
        )
    request_identity = {
        "schema_version": "stage1_synergy_pipeline_input_v2",
        "target_skill_id": req.target_skill_id,
        "readiness_mode": req.readiness_mode,
        "with_residual": req.with_residual,
        "source": source_identity,
        "grouping": grouping_identity,
        "coverage_proxy": proxy_identity,
        "phase_schema": phase_schema_identity,
        "residual_mask": residual_identity,
        "fit": {
            "mode": req.fit_mode,
            "ranks": list(req.ranks),
            "region_ranks": _jsonable(req.region_ranks),
            "total_rank_budget": req.total_rank_budget,
            "require_dynamic_coverage": req.require_dynamic_coverage,
            "max_mean_dynamic_gap": req.max_mean_dynamic_gap,
            "max_key_phase_dynamic_gap": req.max_key_phase_dynamic_gap,
            "max_basis_condition_number": req.max_basis_condition_number,
            "min_effective_rank_fraction": req.min_effective_rank_fraction,
            "expected_environment_fingerprint": (req.expected_environment_fingerprint),
            "expected_rollout_manifest_fingerprint": (req.expected_rollout_manifest_fingerprint),
            "dynamic_coverage_reports": _jsonable(req.dynamic_coverage_reports),
            "seeds": list(req.seeds),
            "normalization": req.normalization,
            "near_zero_threshold": req.near_zero_threshold,
            "phase_weights": phase_weights_identity,
        },
        "config_contract": config_contract,
        "env_prefix": req.env_prefix,
    }
    input_fingerprint = _json_sha256(request_identity)
    object_dir = Path(req.output_root).resolve() / ".objects" / input_fingerprint
    expected_steps = [
        "validate_primitive_source",
        "build_primitive_source_manifest",
        "fit_formal_basis_and_coefficient_statistics",
    ]
    if req.require_dynamic_coverage:
        expected_steps.append("persist_or_validate_dynamic_coverage_candidates")
    if req.readiness_mode == "formal" and proxy_identity is not None:
        expected_steps.append("evaluate_target_static_coverage")
    if req.with_residual:
        expected_steps.append("fit_structured_residual")
    expected_steps.extend(
        [
            "offline_action_interface_preflight",
            "atomic_release_and_ready_publication",
        ]
    )
    can_apply = not missing and source_identity is not None
    return {
        "schema_version": PIPELINE_PLAN_SCHEMA_VERSION,
        "plan_only": True,
        "writes_performed": False,
        "can_apply": can_apply,
        "requested_mode": req.readiness_mode,
        "fit_mode": req.fit_mode,
        "requested_terminal_readiness": ("training_ready_sr" if req.with_residual else "training_ready_s"),
        "input_fingerprint": input_fingerprint,
        "object_dir": str(object_dir),
        "request": asdict(req),
        "request_identity": request_identity,
        "runtime_contract_available": runtime_contract is not None,
        "missing_inputs": missing,
        "warnings": warnings,
        "steps": expected_steps,
    }


def apply_stage1_pipeline(request: PipelineRequest) -> dict[str, Any]:
    """Build an immutable object and atomically publish its achieved readiness."""

    req = _validate_request(request)
    plan = plan_stage1_pipeline(req)
    if not plan["can_apply"]:
        raise PipelineInputError("pipeline inputs are incomplete: " + ", ".join(plan["missing_inputs"]))
    object_dir = Path(plan["object_dir"])
    ready_path = object_dir / "READY.json"
    if object_dir.exists():
        if ready_path.is_file():
            return _load_completed_object(
                object_dir,
                request=req,
                plan=plan,
            )
        quarantined = _quarantine_incomplete_object(
            object_dir,
            output_root=Path(req.output_root).resolve(),
        )
        if quarantined is None:
            return _load_completed_object(
                object_dir,
                request=req,
                plan=plan,
            )
    object_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        object_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        if ready_path.is_file():
            return _load_completed_object(
                object_dir,
                request=req,
                plan=plan,
            )
        raise IncompletePipelineObjectError(
            "a concurrent builder created the same object without committing READY"
        ) from exc
    _atomic_write_json(object_dir / "plan.json", plan)

    source = _materialize_primitive_source(req, object_dir)
    _validate_plan_runtime_contract(
        request=req,
        config_contract=dict(plan["request_identity"]["config_contract"]),
        source_identity=dict(plan["request_identity"]["source"]),
        runtime_contract=source.metadata,
        phase_schema_identity=dict(plan["request_identity"]["phase_schema"]),
    )
    if req.coverage_proxy_artifact is not None:
        proxy_preflight = _load_coverage_proxy_artifact(
            req.coverage_proxy_artifact,
            expected_target_skill_id=req.target_skill_id,
            expected_runtime_contract=source.metadata,
            expected_phase_schema_fingerprint=str(plan["request_identity"]["phase_schema"]["semantic_fingerprint"]),
        )
        _validate_formal_proxy_phase_contract(
            proxy_preflight,
            dict(plan["request_identity"]["config_contract"]),
        )
    # External source/controller/proxy content can change during a long plan or
    # catalog ingest.  Never publish such a build under the old object identity.
    current_plan = plan_stage1_pipeline(req)
    if current_plan["input_fingerprint"] != plan["input_fingerprint"]:
        raise PipelineInputError(
            "pipeline input content changed after planning; the incomplete object was not published"
        )
    fit_config = _fit_config_for_request(req)
    excluded_paths = _expected_target_exclusions(
        (req.bootstrap_config_name if req.readiness_mode == "bootstrap" else req.formal_config_name),
        env_prefix=req.env_prefix,
        readiness_mode=req.readiness_mode,
    )
    checkpoints = _source_checkpoint_fingerprints(source)
    source_manifest_path = object_dir / "source_manifest.json"
    source_manifest = save_primitive_source_manifest_from_splits(
        source_manifest_path,
        train_source=source.train_source,
        validation_source=source.validation_source,
        target_skill_id=req.target_skill_id,
        excluded_target_motion_paths=excluded_paths,
        source_checkpoint_fingerprints=checkpoints,
        fit_config=fit_config,
    )
    source_artifact = {
        "path": str(source_manifest.path.resolve()),
        "fingerprint": source_manifest.fingerprint,
        "dataset_fingerprint": source_manifest.manifest["source_dataset_fingerprint"],
        "train_source": str(source.train_source.resolve()),
        "validation_source": str(source.validation_source.resolve()),
        "source_identity": source.source_identity,
    }
    achieved = "source_validated"
    artifacts: dict[str, Any] = {"source": source_artifact}
    failures: list[dict[str, str]] = []

    fit_output = object_dir / "fit"
    try:
        fit_report = fit_synergy_dataset(
            source.train_source,
            source.validation_source,
            output_dir=fit_output,
            signal_kinds=(EXCITATION_SIGNAL_KIND,),
            mode=req.fit_mode,
            grouping_json=source.regional_grouping_path,
            primitive_source_manifest=source_manifest.path,
            config=fit_config,
            dynamic_coverage_reports=req.dynamic_coverage_reports,
        )
    except BasisNotEligibleForEarlyControl as exc:
        candidate_inventory_path = fit_output / "dynamic_coverage_candidate_inventory.json"
        if candidate_inventory_path.is_file():
            candidate_inventory = load_json_strict(candidate_inventory_path)
            if not isinstance(candidate_inventory, Mapping):
                raise PipelineInputError("dynamic coverage candidate inventory must contain an object") from exc
            expected_inventory_fingerprint = _json_sha256(
                {str(key): value for key, value in candidate_inventory.items() if key != "inventory_fingerprint"}
            )
            if candidate_inventory.get("inventory_fingerprint") != expected_inventory_fingerprint:
                raise PipelineInputError("dynamic coverage candidate inventory fingerprint mismatch") from exc
            artifacts["dynamic_coverage_candidates"] = {
                "path": str(candidate_inventory_path.resolve()),
                "fingerprint": _require_sha256(
                    expected_inventory_fingerprint,
                    "dynamic coverage candidate inventory fingerprint",
                ),
                "status": candidate_inventory.get("status"),
            }
        failures.append({"stage": "basis_fit_gate", "message": str(exc)})
        return _finalize_object(
            req,
            plan,
            object_dir,
            achieved_readiness=achieved,
            artifacts=artifacts,
            bindings=None,
            offline_preflight=None,
            failures=failures,
        )

    preferred = fit_report.get("preferred_decoder_artifacts", {}).get(EXCITATION_SIGNAL_KIND)
    if not isinstance(preferred, Mapping):
        raise RuntimeError("synergy fit did not declare a preferred physical-excitation decoder")
    basis = load_synergy_basis(str(preferred["artifact_path"]))
    if basis.fingerprint != str(preferred["artifact_fingerprint"]):
        raise RuntimeError("preferred formal basis fingerprint changed after fitting")
    stats_path = basis.path / "coefficient_stats.npz"
    stats = load_coefficient_statistics(
        stats_path,
        expected_basis_fingerprint=basis.fingerprint,
        expected_rank=basis.basis.shape[1],
    )
    artifacts["basis"] = {
        "path": str(basis.path.resolve()),
        "fingerprint": basis.fingerprint,
        "rank": int(basis.basis.shape[1]),
        "fit_report_path": str((fit_output / "fit_report.json").resolve()),
    }
    artifacts["coefficient_statistics"] = {
        "path": str(stats_path.resolve()),
        "fingerprint": stats["stats_fingerprint"],
    }
    achieved = "basis_ready"

    residual_artifact: dict[str, Any] | None = None
    if req.with_residual:
        residual_config = _residual_fit_config(req.residual_config_name, req.env_prefix)
        try:
            residual_report = fit_structured_residual_basis(
                source.train_source,
                source.validation_source,
                primary_basis_path=basis.path,
                coefficient_statistics_path=stats_path,
                primitive_source_manifest_path=source_manifest.path,
                expected_primitive_source_manifest_fingerprint=(source_manifest.fingerprint),
                residual_mask_path=_required_path(req.residual_mask, "residual mask"),
                output_path=object_dir / "residual_basis",
                config=residual_config,
            )
            residual_artifact = {
                "path": str(Path(residual_report["artifact_path"]).resolve()),
                "fingerprint": str(residual_report["artifact_fingerprint"]),
                "fit_report_path": str(residual_report["fit_report_path"]),
            }
            artifacts["residual"] = residual_artifact
        except ValueError as exc:
            failures.append({"stage": "residual_fit_gate", "message": str(exc)})

    coverage_artifact: dict[str, Any] | None = None
    if req.readiness_mode == "formal" and req.coverage_proxy_artifact is not None:
        proxy = _load_coverage_proxy_artifact(
            req.coverage_proxy_artifact,
            expected_target_skill_id=req.target_skill_id,
            expected_runtime_contract=source.metadata,
            expected_phase_schema_fingerprint=(
                load_static_proxy_phase_schema(req.phase_schema)["phase_schema_fingerprint"]
            ),
        )
        _validate_formal_proxy_phase_contract(
            proxy,
            dict(plan["request_identity"]["config_contract"]),
        )
        proxy_values, phase_id, proxy_names = _read_proxy_npz(proxy.npz_path)
        upper = 1.2 * np.asarray(stats["coefficient_q99"], dtype=np.float64)
        coverage_thresholds = _coverage_thresholds(
            req.formal_config_name,
            env_prefix=req.env_prefix,
        )
        coverage_report = evaluate_static_proxy_coverage(
            basis,
            proxy_values,
            phase_id=phase_id,
            phase_schema=load_static_proxy_phase_schema(req.phase_schema),
            coefficient_upper_bounds=upper,
            thresholds=coverage_thresholds,
            proxy_muscle_names=proxy_names,
            proxy_producer_binding=proxy.producer_binding,
        )
        gate_path = object_dir / "coverage" / "static_coverage_gate.json"
        stored_gate = write_static_proxy_coverage_gate(gate_path, coverage_report)
        coverage_artifact = {
            "path": str(gate_path.resolve()),
            "fingerprint": str(stored_gate["artifact_fingerprint"]),
            "proxy_fingerprint": str(stored_gate["proxy_fingerprint"]),
            "proxy_artifact_path": str(proxy.manifest_path.resolve()),
            "proxy_artifact_fingerprint": proxy.fingerprint,
            "proxy_npz_path": str(proxy.npz_path.resolve()),
            "passed": bool(stored_gate["passed"]),
        }
        artifacts["coverage"] = coverage_artifact
        if coverage_artifact["passed"]:
            achieved = "coverage_ready"
        else:
            failures.append(
                {
                    "stage": "target_static_coverage_gate",
                    "message": "target-control static coverage thresholds did not pass",
                }
            )

    bindings: dict[str, str] | None = None
    offline_preflight: dict[str, Any] | None = None
    if req.readiness_mode == "bootstrap":
        bindings = _build_binding_variables(
            req,
            source=source_artifact,
            basis=artifacts["basis"],
            stats=artifacts["coefficient_statistics"],
            coverage=None,
            residual=None,
        )
        try:
            offline_preflight = _offline_action_preflight(
                config_name=req.bootstrap_config_name,
                readiness_mode="bootstrap",
                bindings=bindings,
                runtime_contract=source.metadata,
                frozen_decoder_output_path=object_dir / "frozen_body_decoder",
            )
            achieved = "training_ready_s"
        except Exception as exc:
            failures.append({"stage": "bootstrap_offline_preflight", "message": str(exc)})
            bindings = None
    elif achieved == "coverage_ready":
        primary_bindings = _build_binding_variables(
            req,
            source=source_artifact,
            basis=artifacts["basis"],
            stats=artifacts["coefficient_statistics"],
            coverage=coverage_artifact,
            residual=None,
        )
        try:
            primary_preflight = _offline_action_preflight(
                config_name=req.formal_config_name,
                readiness_mode="formal",
                bindings=primary_bindings,
                runtime_contract=source.metadata,
                frozen_decoder_output_path=object_dir / "frozen_body_decoder",
            )
            bindings = primary_bindings
            offline_preflight = primary_preflight
            achieved = "training_ready_s"
        except Exception as exc:
            failures.append({"stage": "formal_offline_preflight", "message": str(exc)})
            bindings = None
        if bindings is not None and req.with_residual and residual_artifact is not None:
            residual_bindings = _build_binding_variables(
                req,
                source=source_artifact,
                basis=artifacts["basis"],
                stats=artifacts["coefficient_statistics"],
                coverage=coverage_artifact,
                residual=residual_artifact,
            )
            try:
                residual_preflight = _offline_action_preflight(
                    config_name=req.residual_config_name,
                    readiness_mode="formal",
                    bindings=residual_bindings,
                    runtime_contract=source.metadata,
                    frozen_decoder_output_path=object_dir / "frozen_body_decoder",
                )
                bindings = residual_bindings
                offline_preflight = residual_preflight
                achieved = "training_ready_sr"
            except Exception as exc:
                failures.append({"stage": "residual_offline_preflight", "message": str(exc)})

    if req.readiness_mode == "formal" and req.coverage_proxy_artifact is None:
        failures.append(
            {
                "stage": "formal_target_proxy",
                "message": (
                    "no independently provenanced ChinaJump target-control proxy; formal bindings were not generated"
                ),
            }
        )

    if offline_preflight is not None and bindings is not None:
        frozen_descriptor = dict(offline_preflight["frozen_body_decoder"])
        artifacts["frozen_body_decoder"] = frozen_descriptor
        bindings.update(
            {
                f"{req.env_prefix}_FROZEN_BODY_DECODER": str(frozen_descriptor["path"]),
                f"{req.env_prefix}_FROZEN_BODY_DECODER_FINGERPRINT": str(frozen_descriptor["fingerprint"]),
                f"{req.env_prefix}_BODY_SYNERGY_CONTRACT": str(frozen_descriptor["body_synergy_contract_path"]),
                f"{req.env_prefix}_BODY_SYNERGY_CONTRACT_FINGERPRINT": str(
                    frozen_descriptor["body_synergy_contract_fingerprint"]
                ),
                f"{req.env_prefix}_BODY_SYNERGY_PORTABLE_CORE_FINGERPRINT": str(
                    frozen_descriptor["portable_decoder_core_fingerprint"]
                ),
            }
        )

    return _finalize_object(
        req,
        plan,
        object_dir,
        achieved_readiness=achieved,
        artifacts=artifacts,
        bindings=bindings,
        offline_preflight=offline_preflight,
        failures=failures,
    )


def preflight_stage1_release(
    release_or_pointer: str | Path,
    *,
    config_name: str | None = None,
    real_env_smoke: bool = False,
) -> dict[str, Any]:
    """Revalidate a published release without W&B, checkpoints, or training."""

    release, release_path = load_stage1_release(release_or_pointer)
    _, _, bindings_payload = _load_release_commit(release_path)
    if bindings_payload is None:
        raise PipelineInputError(f"release readiness={release['readiness']!r} has no training bindings")
    variables = {str(key): str(value) for key, value in dict(bindings_payload["variables"]).items()}
    committed_config = str(bindings_payload["config_name"])
    if config_name is not None and str(config_name) != committed_config:
        raise PipelineInputError("preflight config override differs from committed training config")
    selected_config = committed_config
    offline = _offline_action_preflight(
        config_name=selected_config,
        readiness_mode=str(release["release_mode"]),
        bindings=variables,
        runtime_contract=dict(release["runtime_contract"]),
        expected_frozen_decoder=dict(release["artifacts"]["frozen_body_decoder"]),
    )
    if not isinstance(offline, Mapping) or offline.get("status") != "passed":
        raise PipelineInputError("offline action-interface preflight did not pass")
    real_report: dict[str, Any]
    if real_env_smoke:
        real_report = _real_environment_smoke(selected_config, variables)
    else:
        real_report = {
            "requested": False,
            "status": "not_run",
            "reason": "pass --real-env-smoke to construct the actual training environment",
        }
    return {
        "schema_version": PIPELINE_PREFLIGHT_SCHEMA_VERSION,
        "passed": True,
        "release_path": str(release_path),
        "release_fingerprint": release["release_fingerprint"],
        "release_mode": release["release_mode"],
        "readiness": release["readiness"],
        "config_name": selected_config,
        "offline_action_interface": offline,
        "real_environment_smoke": real_report,
        "training_started": False,
        "wandb_started": False,
        "checkpoint_created": False,
    }


def load_stage1_release(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load either a release manifest or its atomic release pointer."""

    source = Path(path).resolve()
    raw = load_json_strict(source)
    if not isinstance(raw, dict):
        raise PipelineInputError("release/pointer JSON must contain an object")
    pointer: dict[str, Any] | None = None
    if raw.get("schema_version") == PIPELINE_POINTER_SCHEMA_VERSION:
        pointer = _validate_self_fingerprint(
            raw,
            fingerprint_field="pointer_fingerprint",
        )
        source = Path(str(pointer["release_path"])).resolve()
    release, ready, _ = _load_release_commit(source)
    if pointer is not None:
        _validate_pointer_commit_binding(pointer, source, release, ready)
    return release, source


def write_shell_bindings(
    path: str | Path,
    *,
    variables: Mapping[str, str],
    bindings_fingerprint: str,
    release_fingerprint: str,
) -> None:
    """Atomically write a source-able, shell-quoted artifact binding file."""

    lines = [
        "# Generated by the Stage-1 synergy pipeline; do not edit.",
        f"# release_fingerprint={_require_sha256(release_fingerprint, 'release fingerprint')}",
        f"# bindings_fingerprint={_require_sha256(bindings_fingerprint, 'bindings fingerprint')}",
    ]
    for key in sorted(variables):
        if _ENV_NAME_RE.fullmatch(str(key)) is None:
            raise PipelineInputError(f"invalid environment variable name: {key!r}")
        value = str(variables[key])
        if "\x00" in value or "\n" in value or "\r" in value:
            raise PipelineInputError(f"environment variable {key} contains a forbidden control character")
        lines.append(f"export {key}={shlex.quote(value)}")
    _atomic_write_text(Path(path), "\n".join(lines) + "\n")


def _validate_request(request: PipelineRequest) -> PipelineRequest:
    if request.readiness_mode not in {"formal", "bootstrap"}:
        raise PipelineInputError("readiness_mode must be formal or bootstrap")
    if not isinstance(request.fit_mode, str) or request.fit_mode not in {"global", "regional", "both"}:
        raise PipelineInputError("fit_mode must be global, regional, or both")
    if bool(request.primitive_catalog) == bool(request.train or request.val):
        raise PipelineInputError("provide exactly one primitive_catalog or a train+val source pair")
    if request.primitive_catalog is None and (not request.train or not request.val):
        raise PipelineInputError("prepared primitive input requires both train and val")
    if request.readiness_mode == "bootstrap" and request.coverage_proxy_artifact:
        raise PipelineInputError("bootstrap mode must not consume or imply formal target coverage")
    if request.readiness_mode == "bootstrap" and request.with_residual:
        raise PipelineInputError("bootstrap currently has no separate residual Hydra contract")
    if request.readiness_mode == "bootstrap" and (
        request.bootstrap_config_name in {request.formal_config_name, request.residual_config_name}
    ):
        raise PipelineInputError("bootstrap requires a distinct bootstrap-only config")
    if not request.target_skill_id.strip():
        raise PipelineInputError("target_skill_id must be non-empty")
    if _ENV_NAME_RE.fullmatch(request.env_prefix) is None:
        raise PipelineInputError("env_prefix must be an uppercase shell identifier")
    if request.dynamic_coverage_reports is not None and not isinstance(
        request.dynamic_coverage_reports,
        Mapping,
    ):
        raise PipelineInputError("dynamic_coverage_reports must be keyed by signal kind, region, then rank")
    config = SynergyFitConfig(
        ranks=request.ranks,
        region_ranks=request.region_ranks,
        total_rank_budget=request.total_rank_budget,
        require_dynamic_coverage=request.require_dynamic_coverage,
        max_mean_dynamic_gap=request.max_mean_dynamic_gap,
        max_key_phase_dynamic_gap=request.max_key_phase_dynamic_gap,
        max_basis_condition_number=request.max_basis_condition_number,
        min_effective_rank_fraction=request.min_effective_rank_fraction,
        expected_environment_fingerprint=request.expected_environment_fingerprint,
        expected_rollout_manifest_fingerprint=(request.expected_rollout_manifest_fingerprint),
        seeds=request.seeds,
        normalization=request.normalization,
        near_zero_threshold=request.near_zero_threshold,
        phase_weights=_load_phase_weights(request.phase_weights_json),
    ).validated()
    return PipelineRequest(
        **{
            **asdict(request),
            "ranks": tuple(config.ranks),
            "region_ranks": config.region_ranks,
            "total_rank_budget": config.total_rank_budget,
            "require_dynamic_coverage": config.require_dynamic_coverage,
            "max_mean_dynamic_gap": config.max_mean_dynamic_gap,
            "max_key_phase_dynamic_gap": config.max_key_phase_dynamic_gap,
            "max_basis_condition_number": config.max_basis_condition_number,
            "min_effective_rank_fraction": config.min_effective_rank_fraction,
            "expected_environment_fingerprint": (config.expected_environment_fingerprint),
            "expected_rollout_manifest_fingerprint": (config.expected_rollout_manifest_fingerprint),
            "dynamic_coverage_reports": (
                None if request.dynamic_coverage_reports is None else _jsonable(request.dynamic_coverage_reports)
            ),
            "seeds": tuple(config.seeds),
            "normalization": config.normalization,
            "near_zero_threshold": config.near_zero_threshold,
        }
    )


def _materialize_primitive_source(
    request: PipelineRequest,
    object_dir: Path,
) -> PrimitiveSourceView:
    if request.primitive_catalog is not None:
        try:
            from musclemimic.synergy.primitive_ingest import ingest_primitive_catalog
        except ImportError as exc:  # pragma: no cover - transition compatibility
            raise PipelineInputError("primitive catalog input requires musclemimic.synergy.primitive_ingest") from exc
        result = ingest_primitive_catalog(
            request.primitive_catalog,
            object_dir / "primitive_rollouts",
        )
        source_dir = Path(result.output_dir).resolve()
        source_checkpoints = Path(result.source_checkpoints_path).resolve()
        source_identity = {
            "kind": "ingested_primitive_catalog",
            "catalog_path": str(Path(request.primitive_catalog).resolve()),
            "build_fingerprint": _require_sha256(
                result.build_fingerprint,
                "primitive ingest build fingerprint",
            ),
            "dataset_qc_path": str(Path(result.dataset_qc_path).resolve()),
            "idempotent": bool(result.idempotent),
        }
        train_source = validation_source = source_dir
        regional_grouping_path = Path(result.regional_grouping_path).resolve()
    else:
        train_source = Path(_required_path(request.train, "train source")).resolve()
        validation_source = Path(_required_path(request.val, "validation source")).resolve()
        candidate = (
            train_source / "source_checkpoints.json"
            if train_source.is_dir()
            else train_source.parent / "source_checkpoints.json"
        )
        source_checkpoints = candidate if candidate.is_file() else None
        source_identity = {
            "kind": "prepared_primitive_shards",
            "train_path": str(train_source),
            "validation_path": str(validation_source),
        }
        regional_grouping_path = Path(request.grouping_json).resolve()
    train = load_synergy_split(train_source, split="train")
    validation = load_synergy_split(validation_source, split="val")
    metadata = _runtime_contract_from_splits(train, validation)
    source_identity["primitive_semantic_contracts"] = metadata["primitive_semantic_contracts"]
    source_identity["primitive_semantic_attestation"] = metadata["primitive_semantic_attestation"]
    return PrimitiveSourceView(
        train_source=train_source,
        validation_source=validation_source,
        source_checkpoints_path=source_checkpoints,
        regional_grouping_path=regional_grouping_path,
        metadata=metadata,
        source_identity=source_identity,
    )


def _source_checkpoint_fingerprints(source: PrimitiveSourceView) -> dict[str, str]:
    metadata_value = source.metadata.get("source_checkpoint_fingerprints")
    if not isinstance(metadata_value, Mapping) or not metadata_value:
        raise PipelineInputError("primitive metadata lacks source_checkpoint_fingerprints")
    metadata_checkpoints = {
        str(key): _require_sha256(value, f"checkpoint fingerprint for {key}") for key, value in metadata_value.items()
    }
    if source.source_checkpoints_path is not None:
        raw = load_json_strict(source.source_checkpoints_path)
        if not isinstance(raw, Mapping):
            raise PipelineInputError("source_checkpoints.json must contain an object")
        file_checkpoints = {str(key): str(value) for key, value in raw.items()}
        if file_checkpoints != metadata_checkpoints:
            raise PipelineInputError("source_checkpoints.json differs from primitive metadata")
    return metadata_checkpoints


def _fit_config_for_request(request: PipelineRequest) -> SynergyFitConfig:
    config_name = (
        request.bootstrap_config_name
        if request.readiness_mode == "bootstrap"
        else request.residual_config_name
        if request.with_residual
        else request.formal_config_name
    )
    contract = _pipeline_config_contract(
        config_name,
        env_prefix=request.env_prefix,
        readiness_mode=request.readiness_mode,
    )
    thresholds = contract["selection_thresholds"]
    fit_thresholds = {
        "max_basis_condition_number": request.max_basis_condition_number,
        "min_effective_rank_fraction": request.min_effective_rank_fraction,
        **{key: float(value) for key, value in thresholds.items()},
    }
    return SynergyFitConfig(
        ranks=request.ranks,
        region_ranks=request.region_ranks,
        total_rank_budget=request.total_rank_budget,
        require_dynamic_coverage=request.require_dynamic_coverage,
        max_mean_dynamic_gap=request.max_mean_dynamic_gap,
        max_key_phase_dynamic_gap=request.max_key_phase_dynamic_gap,
        expected_environment_fingerprint=request.expected_environment_fingerprint,
        expected_rollout_manifest_fingerprint=(request.expected_rollout_manifest_fingerprint),
        seeds=request.seeds,
        normalization=request.normalization,
        near_zero_threshold=request.near_zero_threshold,
        phase_weights=_load_phase_weights(request.phase_weights_json),
        **fit_thresholds,
    ).validated()


def _coverage_thresholds(
    config_name: str,
    *,
    env_prefix: str,
) -> StaticProxyCoverageThresholds:
    contract = _pipeline_config_contract(
        config_name,
        env_prefix=env_prefix,
        readiness_mode="formal",
    )
    return StaticProxyCoverageThresholds.from_mapping(contract["coverage_thresholds"])


def _residual_fit_config(
    config_name: str,
    env_prefix: str,
) -> StructuredResidualFitConfig:
    cfg = _compose_config(config_name, {}, clear_env_prefix=env_prefix)
    action = cfg.experiment.action_representation
    residual = action.residual
    thresholds = residual.required_fit_thresholds
    return StructuredResidualFitConfig(
        alpha=float(residual.alpha),
        min_dimension=int(residual.min_dimension),
        max_dimension=int(residual.max_dimension),
        max_row_l1_norm=float(residual.max_row_l1_norm),
        min_validation_residual_energy_reduction=float(thresholds.min_validation_residual_energy_reduction),
        min_group_validation_residual_energy_reduction=float(thresholds.min_group_validation_residual_energy_reduction),
        max_validation_coordinate_saturation_fraction=float(thresholds.max_validation_coordinate_saturation_fraction),
    ).validated()


def _pipeline_config_contract(
    config_name: str,
    *,
    env_prefix: str,
    readiness_mode: str,
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = _compose_config(config_name, {}, clear_env_prefix=env_prefix)
    action = cfg.experiment.get("action_representation")
    if action is None or action.get("enabled", False) is not True:
        raise PipelineInputError(f"config {config_name!r} does not enable an action representation")
    if readiness_mode == "bootstrap":
        if (
            cfg.config_status.get("readiness") != "bootstrap_only"
            or action.get("bootstrap_without_target_coverage") is not True
            or action.get("require_coverage_gate") is not False
        ):
            raise PipelineInputError(
                "bootstrap config must declare bootstrap_only, explicitly disable "
                "target coverage, and record bootstrap_without_target_coverage=true"
            )
    elif action.get("require_coverage_gate") is not True:
        raise PipelineInputError("formal config must require a coverage gate")
    return {
        "config_name": config_name,
        "readiness_mode": readiness_mode,
        "expected_target_skill_id": str(action.expected_target_skill_id),
        "expected_underlying_action_dim": int(action.expected_underlying_action_dim),
        "expected_actuator_schema_hash": str(action.expected_actuator_schema_hash),
        "expected_excluded_target_motion_paths": list(action.expected_excluded_target_motion_paths),
        "selection_thresholds": OmegaConf.to_container(
            action.required_selection_thresholds,
            resolve=True,
        ),
        "coverage_thresholds": OmegaConf.to_container(
            action.required_coverage_thresholds,
            resolve=True,
        ),
        "phase_schema_fingerprint": str(action.required_coverage_phase_schema_fingerprint),
        "max_policy_action_dim": int(action.max_policy_action_dim),
        "config_status": OmegaConf.to_container(cfg.config_status, resolve=True),
    }


def _validate_plan_runtime_contract(
    *,
    request: PipelineRequest,
    config_contract: Mapping[str, Any],
    source_identity: Mapping[str, Any] | None,
    runtime_contract: Mapping[str, Any] | None,
    phase_schema_identity: Mapping[str, Any] | None,
) -> None:
    """Fail planning before expensive fitting when the frozen ABIs disagree."""

    if str(config_contract.get("expected_target_skill_id", "")) != request.target_skill_id:
        raise PipelineInputError("training config target skill differs from pipeline target")
    if phase_schema_identity is not None and (
        str(config_contract.get("phase_schema_fingerprint", ""))
        != str(phase_schema_identity.get("semantic_fingerprint", ""))
    ):
        raise PipelineInputError("training config coverage phase schema differs from pipeline schema")

    expected_dim = int(config_contract.get("expected_underlying_action_dim", 0))
    expected_schema = str(config_contract.get("expected_actuator_schema_hash", ""))
    if runtime_contract is not None:
        if len(tuple(runtime_contract.get("actuator_names", ()))) != expected_dim:
            raise PipelineInputError("primitive actuator dimension differs from training config")
        if str(runtime_contract.get("actuator_schema_hash", "")) != expected_schema:
            raise PipelineInputError("primitive ordered actuator schema differs from training config")
        planned_semantics: Any = None
        if source_identity is not None and source_identity.get("kind") == "primitive_catalog":
            build_identity = source_identity.get("build_input_identity")
            if isinstance(build_identity, Mapping):
                planned_semantics = build_identity.get("primitive_semantic_contracts")
        elif source_identity is not None:
            planned_semantics = source_identity.get("primitive_semantic_contracts")
        if planned_semantics is not None and planned_semantics != runtime_contract.get("primitive_semantic_contracts"):
            raise PipelineInputError("primitive semantic contract differs between plan and materialized source")
    elif source_identity is not None and source_identity.get("kind") == "primitive_catalog":
        if str(source_identity.get("target_skill_id", "")) != request.target_skill_id:
            raise PipelineInputError("primitive catalog target skill differs from pipeline target")
        if int(source_identity.get("expected_action_dim", 0)) != expected_dim:
            raise PipelineInputError("primitive catalog action dimension differs from training config")


def _expected_target_exclusions(
    config_name: str,
    *,
    env_prefix: str,
    readiness_mode: str,
) -> tuple[str, ...]:
    contract = _pipeline_config_contract(
        config_name,
        env_prefix=env_prefix,
        readiness_mode=readiness_mode,
    )
    return tuple(str(value) for value in contract["expected_excluded_target_motion_paths"])


def _offline_action_preflight(
    *,
    config_name: str,
    readiness_mode: str,
    bindings: Mapping[str, str],
    runtime_contract: Mapping[str, Any],
    frozen_decoder_output_path: str | Path | None = None,
    expected_frozen_decoder: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _compose_config(
        config_name,
        bindings,
        clear_env_prefix=_common_binding_prefix(bindings),
    )
    action = cfg.experiment.action_representation
    if readiness_mode == "bootstrap":
        if (
            cfg.config_status.get("readiness") != "bootstrap_only"
            or action.get("bootstrap_without_target_coverage") is not True
            or action.get("require_coverage_gate") is not False
        ):
            raise PipelineInputError("selected preflight config is not bootstrap-only")
    else:
        if action.get("bootstrap_without_target_coverage", False) is True:
            raise PipelineInputError("formal preflight cannot use a bootstrap config")
        if action.get("require_coverage_gate") is not True:
            raise PipelineInputError("formal preflight config disabled target coverage")
    names = tuple(str(value) for value in runtime_contract.get("actuator_names", ()))
    ctrlrange = np.asarray(runtime_contract.get("actuator_ctrlrange"), dtype=np.float64)
    model_hash = _require_sha256(
        runtime_contract.get("model_hash", runtime_contract.get("model_fingerprint")),
        "runtime model hash",
    )
    interface = build_early_synergy_action_interface(
        action,
        expected_actuator_names=names,
        runtime_ctrlrange=ctrlrange,
        runtime_model_hash=model_hash,
    )
    if frozen_decoder_output_path is not None and expected_frozen_decoder is not None:
        raise PipelineInputError("offline preflight cannot both export and validate a frozen decoder")
    if frozen_decoder_output_path is not None:
        frozen_path = interface.frozen_decoder.save(frozen_decoder_output_path)
    elif expected_frozen_decoder is not None:
        frozen_path = Path(str(expected_frozen_decoder.get("path", ""))).resolve()
        loaded = load_frozen_body_decoder(
            frozen_path,
            expected_actuator_names=names,
            expected_artifact_fingerprint=str(expected_frozen_decoder.get("fingerprint", "")),
            expected_portable_decoder_core_fingerprint=str(
                expected_frozen_decoder.get("portable_decoder_core_fingerprint", "")
            ),
        )
        if loaded.artifact_fingerprint != interface.frozen_decoder.artifact_fingerprint:
            raise PipelineInputError("released frozen decoder differs from rebuilt Stage-1 action interface")
        interface.body_synergy_contract.assert_exact_runtime_compatible(loaded.body_synergy_contract)
    else:
        frozen_path = None
    frozen_descriptor = {
        "path": None if frozen_path is None else str(frozen_path.resolve()),
        "fingerprint": interface.frozen_decoder.artifact_fingerprint,
        "body_synergy_contract_path": (
            None if frozen_path is None else str((frozen_path / "body_synergy_contract.json").resolve())
        ),
        "body_synergy_contract_fingerprint": (interface.body_synergy_contract.contract_fingerprint),
        "portable_decoder_core_fingerprint": (interface.body_synergy_contract.portable_decoder_core_fingerprint),
        "decoder_core_fingerprint": (interface.frozen_decoder.decoder_core_fingerprint),
    }
    if expected_frozen_decoder is not None:
        expected_descriptor = dict(expected_frozen_decoder)
        if frozen_descriptor != expected_descriptor:
            raise PipelineInputError("released frozen decoder descriptor differs from offline reconstruction")
    return {
        "status": "passed",
        "config_name": config_name,
        "readiness_mode": readiness_mode,
        "policy_action_dim": int(interface.policy_action_dim),
        "body_action_dim": int(interface.body_action_dim),
        "basis_rank": int(interface.synergy_dim),
        "residual_dim": int(interface.residual_dim),
        "physical_action_interface_hash": interface.action_manifest["physical_action_interface_hash"],
        "action_manifest": dict(interface.action_manifest),
        "body_synergy_contract": interface.body_synergy_contract.to_manifest(),
        "frozen_body_decoder": frozen_descriptor,
    }


def _real_environment_smoke(
    config_name: str,
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Construct the real env/action wrapper only; never initialize W&B or PPO."""

    from musclemimic.algorithms.common.env_utils import apply_policy_interface_wrappers
    from musclemimic.runner.engine import (
        instantiate_env,
        instantiate_validation_env,
        validate_experiment_config_status,
    )

    cfg = _compose_config(
        config_name,
        bindings,
        clear_env_prefix=_common_binding_prefix(bindings),
    )
    validate_experiment_config_status(cfg)
    train_env = instantiate_env(cfg)
    validation_env = instantiate_validation_env(cfg, share_trajectory=False)
    train_wrapped = apply_policy_interface_wrappers(train_env, cfg.experiment)
    validation_wrapped = (
        None if validation_env is None else apply_policy_interface_wrappers(validation_env, cfg.experiment)
    )
    train_manifest = dict(train_wrapped.action_manifest)
    if validation_wrapped is not None and (dict(validation_wrapped.action_manifest) != train_manifest):
        raise RuntimeError("train/validation real-env action manifests differ")
    return {
        "requested": True,
        "status": "passed",
        "scope": "real_environment_construction_and_action_wrapper",
        "train_policy_action_dim": int(train_wrapped.info.action_space.shape[0]),
        "validation_constructed": validation_wrapped is not None,
        "physical_action_interface_hash": train_manifest["physical_action_interface_hash"],
        "reset_or_step_executed": False,
        "training_started": False,
        "wandb_started": False,
        "checkpoint_created": False,
    }


def _build_binding_variables(
    request: PipelineRequest,
    *,
    source: Mapping[str, Any],
    basis: Mapping[str, Any],
    stats: Mapping[str, Any],
    coverage: Mapping[str, Any] | None,
    residual: Mapping[str, Any] | None,
) -> dict[str, str]:
    prefix = request.env_prefix
    result = {
        f"{prefix}_SYNERGY_BASIS": str(basis["path"]),
        f"{prefix}_SYNERGY_BASIS_FINGERPRINT": str(basis["fingerprint"]),
        f"{prefix}_PRIMITIVE_SOURCE_MANIFEST": str(source["path"]),
        f"{prefix}_PRIMITIVE_SOURCE_FINGERPRINT": str(source["fingerprint"]),
        f"{prefix}_SYNERGY_COEFFICIENT_STATS_FINGERPRINT": str(stats["fingerprint"]),
    }
    if coverage is not None:
        if coverage.get("passed") is not True:
            raise PipelineInputError("cannot bind a failed formal coverage gate")
        result.update(
            {
                f"{prefix}_SYNERGY_COVERAGE_GATE": str(coverage["path"]),
                f"{prefix}_SYNERGY_COVERAGE_GATE_FINGERPRINT": str(coverage["fingerprint"]),
                f"{prefix}_SYNERGY_PROXY_FINGERPRINT": str(coverage["proxy_fingerprint"]),
            }
        )
    if residual is not None:
        result.update(
            {
                f"{prefix}_SYNERGY_RESIDUAL_BASIS": str(residual["path"]),
                f"{prefix}_SYNERGY_RESIDUAL_FINGERPRINT": str(residual["fingerprint"]),
            }
        )
    return result


def _finalize_object(
    request: PipelineRequest,
    plan: Mapping[str, Any],
    object_dir: Path,
    *,
    achieved_readiness: str,
    artifacts: Mapping[str, Any],
    bindings: Mapping[str, str] | None,
    offline_preflight: Mapping[str, Any] | None,
    failures: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    final_plan = plan_stage1_pipeline(request)
    if final_plan["input_fingerprint"] != plan["input_fingerprint"]:
        raise PipelineInputError(
            "pipeline input content changed during artifact construction; the incomplete object was not published"
        )
    if achieved_readiness not in READINESS_ORDER:
        raise RuntimeError(f"unknown achieved readiness {achieved_readiness!r}")
    if bindings is not None and achieved_readiness not in TRAINING_READINESS:
        raise RuntimeError("non-training-ready release cannot contain bindings")
    runtime_contract = _release_runtime_contract(artifacts)
    config_name = (
        request.bootstrap_config_name
        if request.readiness_mode == "bootstrap"
        else request.residual_config_name
        if achieved_readiness == "training_ready_sr"
        else request.formal_config_name
    )
    limitations = _release_evidence_limitations(request, plan)
    release_unsigned: dict[str, Any] = {
        "schema_version": PIPELINE_RELEASE_SCHEMA_VERSION,
        "input_fingerprint": str(plan["input_fingerprint"]),
        "object_dir": str(object_dir.resolve()),
        "target_skill_id": request.target_skill_id,
        "release_mode": request.readiness_mode,
        "fit_mode": request.fit_mode,
        "readiness": achieved_readiness,
        "readiness_order": READINESS_ORDER[achieved_readiness],
        "ready_for_training": achieved_readiness in TRAINING_READINESS,
        "formal_target_coverage": bool(
            request.readiness_mode == "formal"
            and READINESS_ORDER[achieved_readiness] >= READINESS_ORDER["coverage_ready"]
        ),
        "evidence_limitations": limitations,
        "config_name": config_name,
        "hydra_overrides": list(CANONICAL_HYDRA_OVERRIDES),
        "artifacts": dict(artifacts),
        "runtime_contract": runtime_contract,
        "offline_action_preflight": (None if offline_preflight is None else dict(offline_preflight)),
        "failures": [dict(item) for item in failures],
        "training_bindings": None,
    }
    release_fingerprint = _json_sha256(release_unsigned)
    release = {**release_unsigned, "release_fingerprint": release_fingerprint}
    release_path = object_dir / "release.json"

    binding_descriptor: dict[str, Any] | None = None
    if bindings is not None:
        bindings_unsigned = {
            "schema_version": PIPELINE_BINDINGS_SCHEMA_VERSION,
            "input_fingerprint": str(plan["input_fingerprint"]),
            "target_skill_id": request.target_skill_id,
            "release_fingerprint": release_fingerprint,
            "release_mode": request.readiness_mode,
            "readiness": achieved_readiness,
            "ready_for_training": True,
            "config_name": config_name,
            "hydra_overrides": list(CANONICAL_HYDRA_OVERRIDES),
            "evidence_limitations": limitations,
            "variables": dict(bindings),
        }
        bindings_payload = {
            **bindings_unsigned,
            "bindings_fingerprint": _json_sha256(bindings_unsigned),
        }
        bindings_dir = object_dir / "bindings"
        json_path = bindings_dir / f"{request.readiness_mode}.json"
        shell_path = bindings_dir / f"{request.readiness_mode}.env"
        _atomic_write_json(json_path, bindings_payload)
        write_shell_bindings(
            shell_path,
            variables=bindings,
            bindings_fingerprint=bindings_payload["bindings_fingerprint"],
            release_fingerprint=release_fingerprint,
        )
        shell_sha256 = _file_sha256(shell_path)
        binding_descriptor = {
            "json_path": str(json_path.resolve()),
            "shell_path": str(shell_path.resolve()),
            "fingerprint": bindings_payload["bindings_fingerprint"],
            "shell_sha256": shell_sha256,
            "config_name": config_name,
        }
        # ``training_bindings`` cannot be included in the release's own hash
        # without a release<->bindings fingerprint cycle.  Store the descriptor
        # in a separately self-fingerprinted READY commit record instead.
    _atomic_write_json(release_path, release)
    ready_unsigned = {
        "schema_version": PIPELINE_READY_SCHEMA_VERSION,
        "input_fingerprint": str(plan["input_fingerprint"]),
        "target_skill_id": request.target_skill_id,
        "release_path": str(release_path.resolve()),
        "release_fingerprint": release_fingerprint,
        "release_mode": request.readiness_mode,
        "readiness": achieved_readiness,
        "ready_for_training": achieved_readiness in TRAINING_READINESS,
        "config_name": config_name,
        "hydra_overrides": list(CANONICAL_HYDRA_OVERRIDES),
        "evidence_limitations": limitations,
        "training_bindings": binding_descriptor,
    }
    ready = {**ready_unsigned, "ready_fingerprint": _json_sha256(ready_unsigned)}
    _atomic_write_json(object_dir / "READY.json", ready)

    committed_release, committed_ready, _ = _load_release_commit(
        release_path,
        expected_input_fingerprint=str(plan["input_fingerprint"]),
        expected_mode=request.readiness_mode,
        expected_fit_mode=request.fit_mode,
        expected_target_skill_id=request.target_skill_id,
        expected_config_name=config_name,
    )
    pointer_path = _publish_release_pointer(
        output_root=Path(request.output_root).resolve(),
        target_skill_id=request.target_skill_id,
        release_path=release_path,
        release=committed_release,
        ready=committed_ready,
    )
    result = {
        **release,
        "release_path": str(release_path.resolve()),
        "ready_path": str((object_dir / "READY.json").resolve()),
        "release_pointer_path": str(pointer_path.resolve()),
        "training_bindings": binding_descriptor,
    }
    return result


def _release_evidence_limitations(
    request: PipelineRequest,
    plan: Mapping[str, Any],
) -> list[str]:
    """Bind bootstrap limitations from the already resolved config contract."""

    if request.readiness_mode != "bootstrap":
        return []
    limitations = list(BOOTSTRAP_EVIDENCE_LIMITATIONS)
    request_identity = plan.get("request_identity")
    config_contract = request_identity.get("config_contract") if isinstance(request_identity, Mapping) else None
    config_status = config_contract.get("config_status") if isinstance(config_contract, Mapping) else None
    configured = config_status.get("evidence_limitations", []) if isinstance(config_status, Mapping) else []
    if not isinstance(configured, list) or any(not isinstance(item, str) or not item.strip() for item in configured):
        raise PipelineInputError("config_status.evidence_limitations must be a list of non-empty strings")
    for item in configured:
        normalized = item.strip()
        if normalized not in limitations:
            limitations.append(normalized)
    return limitations


def _load_completed_object(
    object_dir: Path,
    *,
    request: PipelineRequest,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    release_path = object_dir / "release.json"
    expected_config = (
        request.bootstrap_config_name
        if request.readiness_mode == "bootstrap"
        else request.residual_config_name
        if _peek_release_readiness(release_path) == "training_ready_sr"
        else request.formal_config_name
    )
    release, ready, _ = _load_release_commit(
        release_path,
        expected_input_fingerprint=str(plan["input_fingerprint"]),
        expected_mode=request.readiness_mode,
        expected_fit_mode=request.fit_mode,
        expected_target_skill_id=request.target_skill_id,
        expected_config_name=expected_config,
    )
    pointer_path = _publish_release_pointer(
        output_root=Path(request.output_root).resolve(),
        target_skill_id=request.target_skill_id,
        release_path=release_path,
        release=release,
        ready=ready,
    )
    return {
        **release,
        "release_path": str(release_path.resolve()),
        "ready_path": str((object_dir / "READY.json").resolve()),
        "release_pointer_path": str(pointer_path.resolve()),
        "training_bindings": ready.get("training_bindings"),
        "idempotent_reuse": True,
    }


def _quarantine_incomplete_object(
    object_dir: Path,
    *,
    output_root: Path,
) -> Path | None:
    """Atomically preserve an uncommitted object before rebuilding its hash."""

    if (object_dir / "READY.json").is_file():
        return None
    if not object_dir.exists() and not object_dir.is_symlink():
        raise IncompletePipelineObjectError(f"incomplete pipeline object disappeared before quarantine: {object_dir}")
    failed_root = output_root / ".failed"
    failed_root.mkdir(parents=True, exist_ok=True)
    identity = object_dir.name
    for attempt in range(100):
        destination = failed_root / (
            f"{identity}.{time.time_ns()}.pid{os.getpid()}" + ("" if attempt == 0 else f".{attempt}")
        )
        try:
            if (object_dir / "READY.json").is_file():
                return None
            object_dir.rename(destination)
            return destination
        except FileExistsError:
            continue
        except FileNotFoundError as exc:
            raise IncompletePipelineObjectError(
                "incomplete pipeline object changed concurrently during quarantine"
            ) from exc
    raise IncompletePipelineObjectError(f"could not allocate a collision-free quarantine path under {failed_root}")


def _peek_release_readiness(release_path: Path) -> str:
    release = _load_self_fingerprinted_json(
        release_path,
        schema_version=PIPELINE_RELEASE_SCHEMA_VERSION,
        fingerprint_field="release_fingerprint",
    )
    return str(release.get("readiness", ""))


def _release_pointer_path(
    *,
    output_root: Path,
    target_skill_id: str,
    release_mode: str,
    input_fingerprint: str,
) -> Path:
    return output_root / "releases" / (f"{_safe_slug(target_skill_id)}-{release_mode}-{input_fingerprint[:16]}.json")


def _publish_release_pointer(
    *,
    output_root: Path,
    target_skill_id: str,
    release_path: Path,
    release: Mapping[str, Any],
    ready: Mapping[str, Any],
) -> Path:
    """Idempotently publish the deterministic pointer after READY commits."""

    if str(release.get("target_skill_id", "")) != target_skill_id:
        raise PipelineInputError("pointer target skill differs from committed release")
    pointer_path = _release_pointer_path(
        output_root=output_root,
        target_skill_id=target_skill_id,
        release_mode=str(release["release_mode"]),
        input_fingerprint=str(release["input_fingerprint"]),
    )
    pointer_unsigned = {
        "schema_version": PIPELINE_POINTER_SCHEMA_VERSION,
        "input_fingerprint": str(release["input_fingerprint"]),
        "target_skill_id": str(release["target_skill_id"]),
        "release_path": str(release_path.resolve()),
        "release_fingerprint": str(release["release_fingerprint"]),
        "ready_path": str((release_path.parent / "READY.json").resolve()),
        "ready_fingerprint": str(ready["ready_fingerprint"]),
        "release_mode": str(release["release_mode"]),
        "readiness": str(release["readiness"]),
        "ready_for_training": bool(release["ready_for_training"]),
    }
    pointer = {
        **pointer_unsigned,
        "pointer_fingerprint": _json_sha256(pointer_unsigned),
    }
    if pointer_path.is_symlink():
        raise PipelineInputError("deterministic release pointer path cannot be a symlink")
    if pointer_path.exists():
        existing = _load_self_fingerprinted_json(
            pointer_path,
            schema_version=PIPELINE_POINTER_SCHEMA_VERSION,
            fingerprint_field="pointer_fingerprint",
        )
        if _json_sha256(existing) != _json_sha256(pointer):
            raise PipelineInputError("existing deterministic release pointer differs from committed object")
        _validate_pointer_commit_binding(existing, release_path, release, ready)
        return pointer_path
    _atomic_write_json(pointer_path, pointer)
    stored = _load_self_fingerprinted_json(
        pointer_path,
        schema_version=PIPELINE_POINTER_SCHEMA_VERSION,
        fingerprint_field="pointer_fingerprint",
    )
    if _json_sha256(stored) != _json_sha256(pointer):
        raise PipelineInputError("published release pointer changed at commit boundary")
    _validate_pointer_commit_binding(stored, release_path, release, ready)
    return pointer_path


def _release_runtime_contract(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    source = artifacts.get("source")
    if not isinstance(source, Mapping):
        raise RuntimeError("release lacks source artifact")
    train = load_synergy_split(str(source["train_source"]), split="train")
    validation = load_synergy_split(
        str(source["validation_source"]),
        split="val",
    )
    return _runtime_contract_from_splits(train, validation)


def _runtime_contract_from_splits(train: Any, validation: Any) -> dict[str, Any]:
    if train.muscle_names != validation.muscle_names:
        raise PipelineInputError("train/validation actuator names differ")
    fields = (
        "actuator_ctrlrange",
        "ctrlrange_schema_hash",
        "model_hash",
        "source_checkpoint_fingerprints",
        "source_checkpoint_contents",
        "primitive_required_phase_ids",
        "primitive_phase_schema_fingerprints",
        "physical_signal_semantics",
    )
    result: dict[str, Any] = {"actuator_names": list(train.muscle_names)}
    for field in fields:
        left = train.metadata.get(field)
        right = validation.metadata.get(field)
        if _json_sha256(left) != _json_sha256(right):
            raise PipelineInputError(f"primitive train/validation metadata field {field!r} differs")
        if left is None:
            raise PipelineInputError(f"primitive metadata lacks required {field!r}")
        result[field] = left
    required_phase_inventory = result["primitive_required_phase_ids"]
    if not isinstance(required_phase_inventory, Mapping):
        raise PipelineInputError("primitive required-phase metadata must contain an object")
    task_ids = {str(task_id) for task_id in required_phase_inventory}
    for split_name, split in (("train", train), ("validation", validation)):
        if "task_id" in split.arrays:
            array_task_ids = {str(value) for value in np.unique(np.asarray(split.arrays["task_id"])).tolist()}
            if not array_task_ids.issubset(task_ids):
                raise PipelineInputError(f"primitive {split_name} task samples differ from metadata task inventory")
            task_ids.update(array_task_ids)
    try:
        train_semantics = validate_primitive_semantic_contracts(
            sorted(task_ids),
            train.metadata.get("primitive_semantic_contracts"),
            label="primitive train metadata",
        )
        validation_semantics = validate_primitive_semantic_contracts(
            sorted(task_ids),
            validation.metadata.get("primitive_semantic_contracts"),
            label="primitive validation metadata",
        )
    except ValueError as exc:
        raise PipelineInputError(str(exc)) from exc
    if train_semantics != validation_semantics:
        raise PipelineInputError("primitive train/validation semantic contracts differ")
    result["primitive_semantic_contracts"] = train_semantics
    if primitive_semantic_contracts(sorted(task_ids)):
        if Path(train.source).resolve() != Path(validation.source).resolve():
            raise PipelineInputError("P12 train/validation shards must share one sealed ingest directory")
        try:
            from musclemimic.synergy.primitive_ingest import validate_ingested_primitive_dataset

            semantic_attestation = validate_ingested_primitive_dataset(train.source)
        except (FileNotFoundError, ValueError) as exc:
            raise PipelineInputError(f"P12 prepared primitive attestation is invalid: {exc}") from exc
        if semantic_attestation.get("primitive_semantic_contracts") != train_semantics:
            raise PipelineInputError("P12 prepared primitive attestation contract differs from split metadata")
        result["primitive_semantic_attestation"] = semantic_attestation
    else:
        result["primitive_semantic_attestation"] = {}
    ctrlrange = np.asarray(result["actuator_ctrlrange"], dtype=np.float64)
    if ctrlrange.shape != (len(train.muscle_names), 2):
        raise PipelineInputError("primitive actuator_ctrlrange shape is invalid")
    result["actuator_schema_hash"] = actuator_schema_hash(train.muscle_names)
    # Primitive ingest intentionally seals two different identities: the raw
    # source-field schema (``ctrlrange_schema_hash``) and the affine physical
    # transform ABI used by coverage/action code.  Never compare the former to
    # a coverage proxy's transform hash merely because both mention ctrlrange.
    transform_hash = ctrlrange_schema_hash(train.muscle_names, ctrlrange)
    for split_name, split in (("train", train), ("validation", validation)):
        declared = split.metadata.get("transform_ctrlrange_schema_hash")
        if declared is not None and str(declared) != transform_hash:
            raise PipelineInputError(f"primitive {split_name} transform ctrlrange schema hash is stale")
    result["transform_ctrlrange_schema_hash"] = transform_hash
    return _jsonable(result)


def _load_coverage_proxy_artifact(
    path: str | Path,
    *,
    expected_target_skill_id: str,
    expected_runtime_contract: Mapping[str, Any] | None,
    expected_phase_schema_fingerprint: str | None,
) -> CoverageProxyView:
    """Load the forthcoming public proxy artifact, with a strict file fallback."""

    supplied = Path(path).resolve()
    if supplied.suffix == ".npz":
        raise PipelineInputError("formal coverage requires a sealed proxy artifact, never a bare NPZ")
    loaded: Any | None = None
    try:
        from musclemimic.synergy.coverage_proxy import (
            load_coverage_proxy_artifact,
        )
    except ImportError:
        load_coverage_proxy_artifact = None
    if load_coverage_proxy_artifact is not None:
        loaded = load_coverage_proxy_artifact(supplied)
    validated_by_public_loader = loaded is not None
    producer_binding: dict[str, Any] | None = None
    if loaded is not None:
        manifest_raw = getattr(
            loaded,
            "manifest",
            getattr(loaded, "payload", None),
        )
        manifest_path_raw = getattr(
            loaded,
            "manifest_path",
            getattr(loaded, "path", supplied),
        )
        npz_raw = getattr(
            loaded,
            "npz_path",
            getattr(loaded, "proxy_path", getattr(loaded, "data_path", None)),
        )
        fingerprint_raw = getattr(
            loaded,
            "fingerprint",
            getattr(loaded, "manifest_fingerprint", None),
        )
        if not isinstance(manifest_raw, Mapping) or npz_raw is None:
            raise PipelineInputError("coverage_proxy loader result lacks manifest/payload or npz_path")
        manifest = dict(manifest_raw)
        manifest_path = Path(manifest_path_raw).resolve()
        npz_path = Path(npz_raw).resolve()
        fingerprint = _require_sha256(
            fingerprint_raw or manifest.get("artifact_fingerprint") or manifest.get("manifest_fingerprint"),
            "coverage proxy artifact fingerprint",
        )
        binding_raw = getattr(loaded, "oracle_binding", None)
        if isinstance(binding_raw, Mapping):
            producer_binding = dict(binding_raw)
    else:
        manifest_path = supplied / "proxy_manifest.json" if supplied.is_dir() else supplied
        manifest = load_json_strict(manifest_path)
        if not isinstance(manifest, dict):
            raise PipelineInputError("coverage proxy manifest must contain an object")
        fingerprint_field = next(
            (field for field in ("artifact_fingerprint", "manifest_fingerprint") if field in manifest),
            None,
        )
        if fingerprint_field is None:
            raise PipelineInputError("coverage proxy manifest lacks a self fingerprint")
        fingerprint = _require_sha256(
            manifest[fingerprint_field],
            "coverage proxy artifact fingerprint",
        )
        if _json_sha256({key: value for key, value in manifest.items() if key != fingerprint_field}) != fingerprint:
            raise PipelineInputError("coverage proxy manifest fingerprint mismatch")
        npz_value = _first_nested_value(
            manifest,
            (
                "proxy_npz_path",
                "proxy_npz",
                "npz_path",
                "npz_file",
                "data_file",
            ),
        )
        if npz_value is None and supplied.is_dir():
            candidate = supplied / "static_excitation_proxy.npz"
            npz_value = candidate.name if candidate.is_file() else None
        if npz_value is None:
            raise PipelineInputError("coverage proxy manifest does not name its NPZ")
        npz_path = Path(str(npz_value))
        if not npz_path.is_absolute():
            npz_path = manifest_path.parent / npz_path
        npz_path = npz_path.resolve()

    if not npz_path.is_file():
        raise PipelineInputError(f"coverage proxy NPZ does not exist: {npz_path}")
    expected_npz_sha = _first_nested_value(
        manifest,
        ("proxy_npz_sha256", "npz_sha256", "data_sha256", "file_sha256"),
    )
    if expected_npz_sha is None:
        raise PipelineInputError("coverage proxy artifact does not bind its NPZ SHA256")
    if _require_sha256(expected_npz_sha, "coverage proxy NPZ SHA256") != _file_sha256(npz_path):
        raise PipelineInputError("coverage proxy NPZ SHA256 mismatch")
    # The public loader replays the producer/source QC contract and verifies all
    # sealed hashes before it returns.  Transitional fallback manifests must
    # still carry an explicit passed-QC assertion of their own.
    if not validated_by_public_loader and _proxy_qc_passed(manifest) is not True:
        raise PipelineInputError("coverage proxy artifact has no passed QC contract")
    if producer_binding is None:
        raise PipelineInputError("formal coverage proxy loader did not expose its sealed producer/source binding")
    target = _first_nested_value(manifest, ("target_skill_id",))
    if str(target or "") != expected_target_skill_id:
        raise PipelineInputError("coverage proxy target skill differs from pipeline target")
    phase_fingerprint = _first_nested_value(
        manifest,
        ("phase_schema_fingerprint",),
    )
    if expected_phase_schema_fingerprint is not None and (
        str(phase_fingerprint or "") != expected_phase_schema_fingerprint
    ):
        raise PipelineInputError("coverage proxy phase schema fingerprint differs from pipeline schema")
    source_kind = str(
        _first_nested_value(
            manifest,
            ("source_kind", "producer_kind", "control_source_kind"),
        )
        or ""
    ).strip()
    if not source_kind:
        raise PipelineInputError("coverage proxy artifact lacks source_kind")
    lowered = source_kind.lower()
    if any(token in lowered for token in _FORMAL_PROXY_FORBIDDEN_SOURCE_TOKENS):
        raise PipelineInputError("formal target coverage cannot use primitive or early-synergy circular evidence")
    if expected_runtime_contract is not None:
        _validate_proxy_runtime_binding(manifest, expected_runtime_contract)
    required_phase_ids, min_phase_samples, per_phase_counts = _coverage_proxy_phase_evidence(
        loaded=loaded,
        manifest=manifest,
        producer_binding=producer_binding,
    )
    return CoverageProxyView(
        npz_path=npz_path,
        manifest_path=manifest_path,
        manifest=manifest,
        fingerprint=fingerprint,
        source_kind=source_kind,
        producer_binding=producer_binding,
        required_phase_ids=required_phase_ids,
        min_phase_samples=min_phase_samples,
        per_phase_sample_counts=per_phase_counts,
    )


def _coverage_proxy_phase_evidence(
    *,
    loaded: Any | None,
    manifest: Mapping[str, Any],
    producer_binding: Mapping[str, Any],
) -> tuple[tuple[int, ...], int, dict[int, int]]:
    """Adapt sealed proxy v1/v2 phase fields into one strict formal contract."""

    phase_binding = manifest.get("phase_binding")
    selection = manifest.get("selection")
    phase_mapping = phase_binding if isinstance(phase_binding, Mapping) else {}
    selection_mapping = selection if isinstance(selection, Mapping) else {}
    phase_candidates = [
        producer_binding.get("required_phase_ids"),
        getattr(loaded, "required_phase_ids", None),
        phase_mapping.get("required_phase_ids"),
    ]
    min_candidates = [
        producer_binding.get("min_phase_samples"),
        getattr(loaded, "min_phase_samples", None),
        selection_mapping.get("min_phase_samples"),
        phase_mapping.get("min_phase_samples"),
    ]
    count_candidates = [
        producer_binding.get("per_phase_sample_counts"),
        getattr(loaded, "per_phase_sample_counts", None),
        phase_mapping.get("per_phase_sample_counts"),
    ]
    required = _coalesce_phase_id_candidates(phase_candidates)
    minimum = _coalesce_positive_int_candidates(
        min_candidates,
        label="coverage proxy min_phase_samples",
    )
    counts = _coalesce_phase_count_candidates(count_candidates)
    if not required:
        raise PipelineInputError("formal coverage proxy required_phase_ids is empty")
    for phase in required:
        if counts.get(phase, 0) < minimum:
            raise PipelineInputError(
                "coverage proxy formal phase is below its sealed min_phase_samples: "
                f"phase={phase} count={counts.get(phase, 0)} minimum={minimum}"
            )
    return required, minimum, counts


def _coalesce_phase_id_candidates(values: Sequence[Any]) -> tuple[int, ...]:
    parsed: list[tuple[int, ...]] = []
    for value in values:
        if value is None:
            continue
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            raise PipelineInputError("coverage proxy required_phase_ids must be an integer sequence")
        result: list[int] = []
        for item in value:
            if type(item) is not int or item < 0:
                raise PipelineInputError("coverage proxy required_phase_ids must be non-negative integers")
            result.append(int(item))
        candidate = tuple(result)
        if len(set(candidate)) != len(candidate):
            raise PipelineInputError("coverage proxy required_phase_ids contains duplicates")
        parsed.append(candidate)
    if not parsed:
        raise PipelineInputError("coverage proxy lacks required_phase_ids evidence")
    if any(candidate != parsed[0] for candidate in parsed[1:]):
        raise PipelineInputError("coverage proxy required_phase_ids fields disagree")
    return parsed[0]


def _coalesce_positive_int_candidates(values: Sequence[Any], *, label: str) -> int:
    parsed: list[int] = []
    for value in values:
        if value is None:
            continue
        if type(value) is not int or value <= 0:
            raise PipelineInputError(f"{label} must be a positive integer")
        parsed.append(int(value))
    if not parsed:
        raise PipelineInputError(f"formal artifact lacks {label}")
    if any(value != parsed[0] for value in parsed[1:]):
        raise PipelineInputError(f"{label} fields disagree")
    return parsed[0]


def _coalesce_phase_count_candidates(values: Sequence[Any]) -> dict[int, int]:
    parsed: list[dict[int, int]] = []
    for value in values:
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise PipelineInputError("coverage proxy per_phase_sample_counts must be an object")
        candidate: dict[int, int] = {}
        for raw_key, raw_count in value.items():
            try:
                phase = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise PipelineInputError("coverage proxy phase-count key is not an integer") from exc
            if str(phase) != str(raw_key) and raw_key != phase:
                raise PipelineInputError("coverage proxy phase-count key is not canonical")
            if phase < 0 or type(raw_count) is not int or raw_count < 0:
                raise PipelineInputError("coverage proxy phase counts must be non-negative integers")
            candidate[phase] = int(raw_count)
        parsed.append(candidate)
    if not parsed:
        raise PipelineInputError("coverage proxy lacks per_phase_sample_counts evidence")
    if any(candidate != parsed[0] for candidate in parsed[1:]):
        raise PipelineInputError("coverage proxy per_phase_sample_counts fields disagree")
    return parsed[0]


def _validate_formal_proxy_phase_contract(
    proxy: CoverageProxyView,
    config_contract: Mapping[str, Any],
) -> None:
    thresholds = config_contract.get("coverage_thresholds")
    if not isinstance(thresholds, Mapping):
        raise PipelineInputError("formal training config lacks coverage thresholds")
    raw_required = thresholds.get("required_phase_ids")
    if not isinstance(raw_required, Sequence) or isinstance(raw_required, str | bytes):
        raise PipelineInputError("formal training config lacks required coverage phase ids")
    expected: list[int] = []
    for value in raw_required:
        if type(value) is not int or value < 0:
            raise PipelineInputError("formal config required_phase_ids must be non-negative integers")
        expected.append(int(value))
    expected_ids = tuple(expected)
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise PipelineInputError("formal config required_phase_ids must be non-empty and unique")
    if proxy.required_phase_ids != expected_ids:
        raise PipelineInputError("coverage proxy required_phase_ids differ from the formal training config")
    for phase in expected_ids:
        count = proxy.per_phase_sample_counts.get(phase, 0)
        if count < proxy.min_phase_samples:
            raise PipelineInputError(
                "coverage proxy does not meet min_phase_samples for every formal phase: "
                f"phase={phase} count={count} minimum={proxy.min_phase_samples}"
            )


def _validate_proxy_runtime_binding(
    manifest: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
) -> None:
    expected_model = _require_sha256(
        runtime_contract.get("model_hash", runtime_contract.get("model_fingerprint")),
        "primitive runtime model hash",
    )
    proxy_model = _first_nested_value(manifest, ("model_hash", "model_fingerprint"))
    if str(proxy_model or "") != expected_model:
        raise PipelineInputError("coverage proxy model hash differs from primitive/runtime model")
    expected_actuator = str(
        runtime_contract.get("actuator_schema_hash") or actuator_schema_hash(runtime_contract.get("actuator_names", ()))
    )
    proxy_actuator = _first_nested_value(manifest, ("actuator_schema_hash",))
    if str(proxy_actuator or "") != expected_actuator:
        raise PipelineInputError("coverage proxy actuator schema differs from runtime")
    expected_ctrl = str(
        runtime_contract.get("transform_ctrlrange_schema_hash")
        or ctrlrange_schema_hash(
            tuple(str(value) for value in runtime_contract.get("actuator_names", ())),
            np.asarray(runtime_contract.get("actuator_ctrlrange"), dtype=np.float64),
        )
    )
    proxy_ctrl = _first_nested_value(
        manifest,
        ("ctrlrange_schema_hash", "control_range_hash"),
    )
    if str(proxy_ctrl or "") != expected_ctrl:
        raise PipelineInputError("coverage proxy control-range schema differs from runtime")


def _read_proxy_npz(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as data:
        required = {"physical_excitation", "phase_id", "actuator_names"}
        missing = sorted(required - set(data.files))
        if missing:
            raise PipelineInputError(f"coverage proxy NPZ lacks required fields: {missing}")
        excitation = np.asarray(data["physical_excitation"], dtype=np.float64)
        raw_phase = np.asarray(data["phase_id"])
        names = tuple(str(value) for value in np.asarray(data["actuator_names"]).tolist())
    if excitation.ndim != 2 or excitation.shape[0] <= 0:
        raise PipelineInputError("coverage proxy physical_excitation must be non-empty [T,M]")
    if raw_phase.shape != (excitation.shape[0],):
        raise PipelineInputError("coverage proxy phase_id shape differs from excitation")
    if np.issubdtype(raw_phase.dtype, np.bool_) or not np.issubdtype(
        raw_phase.dtype,
        np.integer,
    ):
        raise PipelineInputError("coverage proxy phase_id must use an integer dtype")
    if len(names) != excitation.shape[1] or len(set(names)) != len(names):
        raise PipelineInputError("coverage proxy actuator_names do not match columns")
    return excitation, raw_phase.astype(np.int64), names


def _proxy_qc_passed(manifest: Mapping[str, Any]) -> bool:
    for key in ("passed", "qc_passed"):
        value = manifest.get(key)
        if type(value) is bool:
            return value
    for key in ("qc", "dataset_qc", "proxy_qc"):
        nested = manifest.get(key)
        if isinstance(nested, Mapping) and type(nested.get("passed")) is bool:
            return bool(nested["passed"])
    return False


def _load_primitive_catalog(path: Path, *, require_build_ready: bool) -> Any:
    try:
        from musclemimic.synergy.primitive_catalog import load_primitive_catalog
    except ImportError as exc:  # pragma: no cover - transition compatibility
        raise PipelineInputError("primitive catalog input requires musclemimic.synergy.primitive_catalog") from exc
    return load_primitive_catalog(path, require_build_ready=require_build_ready)


def _validate_build_ready_catalog(catalog: Any) -> None:
    from musclemimic.synergy.primitive_catalog import validate_build_ready_catalog

    validate_build_ready_catalog(catalog)


def _catalog_build_input_identity(catalog: Any) -> dict[str, Any]:
    model_path = getattr(catalog, "model_xml_path", None)
    grouping_path = getattr(catalog, "regional_grouping_path", None)
    tasks = []
    for task in getattr(catalog, "tasks", ()):
        controller = getattr(task, "controller_artifact", None)
        tasks.append(
            {
                "task_id": str(getattr(task, "task_id", "")),
                "enabled": bool(getattr(task, "enabled", False)),
                "phase_schema_fingerprint": str(getattr(getattr(task, "phase_schema", None), "fingerprint", "")),
                "controller": (None if controller is None else _path_content_identity(Path(controller))),
                "trials": [
                    {
                        "trial_id": str(getattr(trial, "trial_id", "")),
                        "split": str(getattr(trial, "split", "")),
                        "raw_npz": _path_content_identity(Path(trial.raw_npz_path)),
                        "rollout_manifest": _path_content_identity(Path(trial.rollout_manifest_path)),
                    }
                    for trial in getattr(task, "trials", ())
                ],
            }
        )
    enabled_task_ids = [task["task_id"] for task in tasks if task["enabled"]]
    return {
        "model": None if model_path is None else _path_content_identity(Path(model_path)),
        "regional_grouping_path": None if grouping_path is None else str(Path(grouping_path).resolve()),
        "regional_grouping": (None if grouping_path is None else _path_content_identity(Path(grouping_path))),
        "primitive_semantic_contracts": primitive_semantic_contracts(enabled_task_ids),
        "tasks": tasks,
    }


def _compose_config(
    config_name: str,
    bindings: Mapping[str, str],
    *,
    clear_env_prefix: str,
) -> Any:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[2]
    with _temporary_environment(bindings, clear_prefix=clear_env_prefix):
        with initialize_config_dir(
            version_base=None,
            config_dir=str(repo_root / "fullbody"),
        ):
            cfg = compose(
                config_name=config_name.removesuffix(".yaml"),
                overrides=["config_status.allow_nonproduction_runtime=true"],
            )
        OmegaConf.resolve(cfg)
    return cfg


@contextmanager
def _temporary_environment(
    bindings: Mapping[str, str],
    *,
    clear_prefix: str,
):
    keys_to_clear = {
        key
        for key in os.environ
        if key.startswith(f"{clear_prefix}_") and ("SYNERGY" in key or "PRIMITIVE_SOURCE" in key)
    }
    keys = keys_to_clear | set(bindings)
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys_to_clear:
            os.environ.pop(key, None)
        for key, value in bindings.items():
            os.environ[str(key)] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _common_binding_prefix(bindings: Mapping[str, str]) -> str:
    keys = list(bindings)
    if not keys:
        return DEFAULT_ENV_PREFIX
    first = keys[0]
    marker = "_SYNERGY_" if "_SYNERGY_" in first else "_PRIMITIVE_SOURCE_"
    return first.split(marker, 1)[0]


def _load_phase_weights(path: str | None) -> dict[int, float] | None:
    if path is None:
        return None
    payload = load_json_strict(Path(path))
    if not isinstance(payload, Mapping):
        raise PipelineInputError("phase weights JSON must contain an object")
    return {int(key): float(value) for key, value in payload.items()}


def _load_region_ranks_json(
    path: str | None,
) -> Mapping[str, tuple[int, ...]] | None:
    if path is None:
        return None
    payload = load_json_strict(Path(path))
    if not isinstance(payload, Mapping):
        raise PipelineInputError("region ranks JSON must contain an object mapping region to rank list")
    return SynergyFitConfig(region_ranks=payload).validated().region_ranks


def _load_dynamic_coverage_reports_json(
    path: str | None,
) -> Mapping[str, Mapping[str, Mapping[int | str, Mapping[str, Any]]]] | None:
    if path is None:
        return None
    payload = load_json_strict(Path(path))
    if not isinstance(payload, Mapping):
        raise PipelineInputError("dynamic coverage reports JSON must contain an object")
    return _jsonable(payload)


def _file_identity_or_missing(
    path: Path,
    *,
    label: str,
    missing: list[str],
) -> dict[str, Any] | None:
    if not path.is_file():
        missing.append(f"{label}:{path}")
        return None
    return {"path": str(path), "sha256": _file_sha256(path)}


def _path_content_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_file():
        return {
            "kind": "file",
            "path": str(resolved),
            "sha256": _file_sha256(resolved),
        }
    if resolved.is_dir():
        files = [item for item in sorted(resolved.rglob("*")) if item.is_file()]
        return {
            "kind": "directory",
            "path": str(resolved),
            "files": [
                {
                    "path": str(item.relative_to(resolved)),
                    "sha256": _file_sha256(item),
                }
                for item in files
            ],
        }
    return {"kind": "missing", "path": str(resolved)}


def _required_existing_path(
    value: str | None,
    label: str,
    missing: list[str],
) -> Path | None:
    if not value:
        missing.append(label)
        return None
    path = Path(value).resolve()
    if not path.exists():
        missing.append(f"{label}:{path}")
        return None
    return path


def _required_path(value: str | None, label: str) -> str:
    if not value:
        raise PipelineInputError(f"{label} is required")
    return value


def _first_nested_value(value: Any, keys: Sequence[str]) -> Any | None:
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                return value[key]
        for nested in value.values():
            found = _first_nested_value(nested, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _first_nested_value(nested, keys)
            if found is not None:
                return found
    return None


def _safe_slug(value: str) -> str:
    result = _SAFE_SLUG_RE.sub("-", value.strip()).strip("-.")
    if not result:
        raise PipelineInputError("target skill id cannot be converted to a safe slug")
    return result.lower()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _load_self_fingerprinted_json(
    path: Path,
    *,
    schema_version: str,
    fingerprint_field: str,
) -> dict[str, Any]:
    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise PipelineInputError(f"{path} must contain a JSON object")
    if payload.get("schema_version") != schema_version:
        raise PipelineInputError(f"unsupported schema in {path}")
    return _validate_self_fingerprint(payload, fingerprint_field=fingerprint_field)


def _load_release_commit(
    release_path: Path,
    *,
    expected_input_fingerprint: str | None = None,
    expected_mode: str | None = None,
    expected_fit_mode: str | None = None,
    expected_target_skill_id: str | None = None,
    expected_config_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Validate the release -> READY -> JSON/shell binding commit graph."""

    canonical_release_path = release_path.resolve()
    release = _load_self_fingerprinted_json(
        canonical_release_path,
        schema_version=PIPELINE_RELEASE_SCHEMA_VERSION,
        fingerprint_field="release_fingerprint",
    )
    _validate_release_semantics(
        release,
        canonical_release_path,
        expected_input_fingerprint=expected_input_fingerprint,
        expected_mode=expected_mode,
        expected_fit_mode=expected_fit_mode,
        expected_target_skill_id=expected_target_skill_id,
        expected_config_name=expected_config_name,
    )
    ready_path = canonical_release_path.parent / "READY.json"
    ready = _load_self_fingerprinted_json(
        ready_path,
        schema_version=PIPELINE_READY_SCHEMA_VERSION,
        fingerprint_field="ready_fingerprint",
    )
    _validate_ready_semantics(ready, release, canonical_release_path)
    descriptor = ready.get("training_bindings")
    bindings: dict[str, Any] | None = None
    if descriptor is not None:
        if not isinstance(descriptor, Mapping):
            raise PipelineInputError("READY training_bindings must be an object or null")
        bindings = _validate_binding_commit(
            dict(descriptor),
            release=release,
            object_dir=canonical_release_path.parent,
        )
    return release, ready, bindings


def _validate_release_semantics(
    release: Mapping[str, Any],
    release_path: Path,
    *,
    expected_input_fingerprint: str | None,
    expected_mode: str | None,
    expected_fit_mode: str | None,
    expected_target_skill_id: str | None,
    expected_config_name: str | None,
) -> None:
    input_fingerprint = _require_sha256(
        release.get("input_fingerprint"),
        "release input_fingerprint",
    )
    if expected_input_fingerprint is not None and input_fingerprint != _require_sha256(
        expected_input_fingerprint,
        "expected input_fingerprint",
    ):
        raise PipelineInputError("release input fingerprint differs from current plan")
    if Path(str(release.get("object_dir", ""))).resolve() != release_path.parent:
        raise PipelineInputError("release object_dir differs from its committed location")
    mode = str(release.get("release_mode", ""))
    if mode not in {"formal", "bootstrap"}:
        raise PipelineInputError("release mode is invalid")
    # Historical v1 releases predate this field, when the pipeline always
    # invoked the fitter with ``mode="both"``.  Preserve that single safe
    # interpretation while requiring every newly written release to be explicit.
    fit_mode = release.get("fit_mode", "both")
    if fit_mode not in {"global", "regional", "both"}:
        raise PipelineInputError("release fit_mode is invalid")
    if expected_fit_mode is not None and fit_mode != expected_fit_mode:
        raise PipelineInputError("release fit_mode differs from current request")
    if expected_mode is not None and mode != expected_mode:
        raise PipelineInputError("release mode differs from current request")
    target_skill_id = str(release.get("target_skill_id", ""))
    if not target_skill_id:
        raise PipelineInputError("release target_skill_id is empty")
    if expected_target_skill_id is not None and target_skill_id != expected_target_skill_id:
        raise PipelineInputError("release target skill differs from current request")
    readiness = str(release.get("readiness", ""))
    if readiness not in READINESS_ORDER:
        raise PipelineInputError("release readiness is invalid")
    readiness_order = release.get("readiness_order")
    if type(readiness_order) is not int or readiness_order != READINESS_ORDER[readiness]:
        raise PipelineInputError("release readiness_order disagrees with readiness")
    ready_for_training = _strict_pipeline_bool(
        release.get("ready_for_training"),
        "release ready_for_training",
    )
    if ready_for_training != (readiness in TRAINING_READINESS):
        raise PipelineInputError("release ready_for_training disagrees with readiness")
    if mode == "bootstrap" and readiness not in {
        "source_validated",
        "basis_ready",
        "training_ready_s",
    }:
        raise PipelineInputError("bootstrap release has an impossible readiness")
    formal_coverage = _strict_pipeline_bool(
        release.get("formal_target_coverage"),
        "release formal_target_coverage",
    )
    expected_formal_coverage = bool(
        mode == "formal" and READINESS_ORDER[readiness] >= READINESS_ORDER["coverage_ready"]
    )
    if formal_coverage != expected_formal_coverage:
        raise PipelineInputError("release formal coverage claim disagrees with mode/readiness")
    limitations = release.get("evidence_limitations")
    if mode == "formal":
        if limitations != []:
            raise PipelineInputError("formal release evidence limitations disagree with release mode")
    elif (
        not isinstance(limitations, list)
        or any(not isinstance(item, str) or not item.strip() for item in limitations)
        or len(set(limitations)) != len(limitations)
        or limitations[: len(BOOTSTRAP_EVIDENCE_LIMITATIONS)] != list(BOOTSTRAP_EVIDENCE_LIMITATIONS)
    ):
        raise PipelineInputError("bootstrap release evidence limitations are invalid")
    if release.get("hydra_overrides") != list(CANONICAL_HYDRA_OVERRIDES):
        raise PipelineInputError("release Hydra overrides differ from the launch contract")
    config_name = str(release.get("config_name", ""))
    if not config_name:
        raise PipelineInputError("release config_name is empty")
    if expected_config_name is not None and config_name != expected_config_name:
        raise PipelineInputError("release config differs from current request/readiness")
    if release.get("training_bindings") is not None:
        raise PipelineInputError("release must delegate training bindings to READY")
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("source"), Mapping):
        raise PipelineInputError("release lacks its primitive source artifact")
    if READINESS_ORDER[readiness] >= READINESS_ORDER["basis_ready"]:
        for name in ("basis", "coefficient_statistics"):
            if not isinstance(artifacts.get(name), Mapping):
                raise PipelineInputError(f"release readiness requires {name} artifact")
    if formal_coverage:
        coverage = artifacts.get("coverage")
        if not isinstance(coverage, Mapping) or coverage.get("passed") is not True:
            raise PipelineInputError("formal coverage readiness requires a passed coverage artifact")
    elif mode == "bootstrap" and artifacts.get("coverage") is not None:
        raise PipelineInputError("bootstrap release cannot contain formal target coverage")
    if readiness == "training_ready_sr" and not isinstance(
        artifacts.get("residual"),
        Mapping,
    ):
        raise PipelineInputError("training_ready_sr requires a residual artifact")
    offline = release.get("offline_action_preflight")
    if ready_for_training:
        if not isinstance(offline, Mapping) or offline.get("status") != "passed":
            raise PipelineInputError("training-ready release lacks a passed offline preflight")
        frozen = artifacts.get("frozen_body_decoder")
        if not isinstance(frozen, Mapping):
            raise PipelineInputError("training-ready release lacks its portable frozen body decoder")
        if offline.get("frozen_body_decoder") != frozen:
            raise PipelineInputError("release frozen decoder differs from offline preflight")
        if str(offline.get("config_name", "")) != config_name or str(offline.get("readiness_mode", "")) != mode:
            raise PipelineInputError("release offline preflight config/mode binding differs")
    runtime = release.get("runtime_contract")
    if not isinstance(runtime, Mapping):
        raise PipelineInputError("release runtime_contract must be an object")
    current_runtime = _release_runtime_contract(artifacts)
    if current_runtime.get("primitive_semantic_contracts") and (
        runtime.get("primitive_semantic_contracts") != current_runtime["primitive_semantic_contracts"]
        or runtime.get("primitive_semantic_attestation") != current_runtime["primitive_semantic_attestation"]
    ):
        raise PipelineInputError("release P12 semantic attestation differs from its current primitive source")


def _validate_ready_semantics(
    ready: Mapping[str, Any],
    release: Mapping[str, Any],
    release_path: Path,
) -> None:
    ready_for_training = _strict_pipeline_bool(
        ready.get("ready_for_training"),
        "READY ready_for_training",
    )
    if ready_for_training != release["ready_for_training"]:
        raise PipelineInputError("READY ready_for_training differs from release")
    comparisons = {
        "input_fingerprint": release["input_fingerprint"],
        "target_skill_id": release["target_skill_id"],
        "release_fingerprint": release["release_fingerprint"],
        "release_mode": release["release_mode"],
        "readiness": release["readiness"],
        "config_name": release["config_name"],
        "hydra_overrides": release["hydra_overrides"],
        "evidence_limitations": release["evidence_limitations"],
    }
    for field, expected in comparisons.items():
        if ready.get(field) != expected:
            raise PipelineInputError(f"READY {field} differs from release")
    if Path(str(ready.get("release_path", ""))).resolve() != release_path:
        raise PipelineInputError("READY release_path differs from committed release")
    descriptor = ready.get("training_bindings")
    if bool(descriptor is not None) != release["ready_for_training"]:
        raise PipelineInputError("READY training bindings disagree with ready_for_training")


def _validate_binding_commit(
    descriptor: Mapping[str, Any],
    *,
    release: Mapping[str, Any],
    object_dir: Path,
) -> dict[str, Any]:
    mode = str(release["release_mode"])
    expected_json_path = (object_dir / "bindings" / f"{mode}.json").resolve()
    expected_shell_path = (object_dir / "bindings" / f"{mode}.env").resolve()
    json_path = Path(str(descriptor.get("json_path", ""))).resolve()
    shell_path = Path(str(descriptor.get("shell_path", ""))).resolve()
    if json_path != expected_json_path or shell_path != expected_shell_path:
        raise PipelineInputError("READY binding paths are not canonical for this object/mode")
    if str(descriptor.get("config_name", "")) != release["config_name"]:
        raise PipelineInputError("READY binding config differs from release")
    descriptor_fingerprint = _require_sha256(
        descriptor.get("fingerprint"),
        "READY bindings fingerprint",
    )
    expected_shell_sha256 = _require_sha256(
        descriptor.get("shell_sha256"),
        "READY shell binding SHA256",
    )
    bindings = _load_self_fingerprinted_json(
        json_path,
        schema_version=PIPELINE_BINDINGS_SCHEMA_VERSION,
        fingerprint_field="bindings_fingerprint",
    )
    if bindings["bindings_fingerprint"] != descriptor_fingerprint:
        raise PipelineInputError("READY binding fingerprint differs from bindings JSON")
    comparisons = {
        "input_fingerprint": release["input_fingerprint"],
        "target_skill_id": release["target_skill_id"],
        "release_fingerprint": release["release_fingerprint"],
        "release_mode": release["release_mode"],
        "readiness": release["readiness"],
        "config_name": release["config_name"],
        "hydra_overrides": release["hydra_overrides"],
        "evidence_limitations": release["evidence_limitations"],
    }
    for field, expected in comparisons.items():
        if bindings.get(field) != expected:
            raise PipelineInputError(f"bindings {field} differs from release/READY")
    if not _strict_pipeline_bool(
        bindings.get("ready_for_training"),
        "bindings ready_for_training",
    ):
        raise PipelineInputError("training bindings cannot declare not-ready")
    variables = bindings.get("variables")
    if not isinstance(variables, Mapping) or not variables:
        raise PipelineInputError("training bindings variables must be a non-empty object")
    normalized_variables: dict[str, str] = {}
    for key, value in variables.items():
        name = str(key)
        if _ENV_NAME_RE.fullmatch(name) is None or not isinstance(value, str):
            raise PipelineInputError("training binding variables contain an invalid name/value")
        if any(token in value for token in ("\x00", "\n", "\r")):
            raise PipelineInputError("training binding variable contains a forbidden control character")
        normalized_variables[name] = value
    _validate_binding_variable_semantics(
        normalized_variables,
        mode=mode,
        readiness=str(release["readiness"]),
    )
    if not shell_path.is_file():
        raise PipelineInputError("READY shell binding artifact does not exist")
    if _file_sha256(shell_path) != expected_shell_sha256:
        raise PipelineInputError("READY shell binding SHA256 mismatch")
    return bindings


def _validate_binding_variable_semantics(
    variables: Mapping[str, str],
    *,
    mode: str,
    readiness: str,
) -> None:
    base_tails = (
        "_SYNERGY_BASIS",
        "_SYNERGY_BASIS_FINGERPRINT",
        "_PRIMITIVE_SOURCE_MANIFEST",
        "_PRIMITIVE_SOURCE_FINGERPRINT",
        "_SYNERGY_COEFFICIENT_STATS_FINGERPRINT",
        "_FROZEN_BODY_DECODER",
        "_FROZEN_BODY_DECODER_FINGERPRINT",
        "_BODY_SYNERGY_CONTRACT",
        "_BODY_SYNERGY_CONTRACT_FINGERPRINT",
        "_BODY_SYNERGY_PORTABLE_CORE_FINGERPRINT",
    )
    basis_keys = [key for key in variables if key.endswith("_SYNERGY_BASIS")]
    if len(basis_keys) != 1:
        raise PipelineInputError("training bindings must identify exactly one environment prefix")
    prefix = basis_keys[0][: -len("_SYNERGY_BASIS")]
    if _ENV_NAME_RE.fullmatch(prefix) is None:
        raise PipelineInputError("training binding environment prefix is invalid")
    expected_tails = list(base_tails)
    if mode == "formal":
        expected_tails.extend(
            (
                "_SYNERGY_COVERAGE_GATE",
                "_SYNERGY_COVERAGE_GATE_FINGERPRINT",
                "_SYNERGY_PROXY_FINGERPRINT",
            )
        )
    if readiness == "training_ready_sr":
        expected_tails.extend(
            (
                "_SYNERGY_RESIDUAL_BASIS",
                "_SYNERGY_RESIDUAL_FINGERPRINT",
            )
        )
    expected_keys = {f"{prefix}{tail}" for tail in expected_tails}
    if set(variables) != expected_keys:
        raise PipelineInputError("training binding variable set disagrees with mode/readiness")


def _validate_pointer_commit_binding(
    pointer: Mapping[str, Any],
    release_path: Path,
    release: Mapping[str, Any],
    ready: Mapping[str, Any],
) -> None:
    pointer_ready = _strict_pipeline_bool(
        pointer.get("ready_for_training"),
        "pointer ready_for_training",
    )
    if pointer_ready != release["ready_for_training"]:
        raise PipelineInputError("release pointer ready_for_training differs from commit")
    comparisons = {
        "input_fingerprint": release["input_fingerprint"],
        "target_skill_id": release["target_skill_id"],
        "release_fingerprint": release["release_fingerprint"],
        "ready_fingerprint": ready["ready_fingerprint"],
        "release_mode": release["release_mode"],
        "readiness": release["readiness"],
    }
    for field, expected in comparisons.items():
        if pointer.get(field) != expected:
            raise PipelineInputError(f"release pointer {field} differs from commit")
    if Path(str(pointer.get("release_path", ""))).resolve() != release_path.resolve():
        raise PipelineInputError("release pointer path differs from committed release")
    if Path(str(pointer.get("ready_path", ""))).resolve() != (release_path.resolve().parent / "READY.json"):
        raise PipelineInputError("release pointer READY path differs from commit")


def _strict_pipeline_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise PipelineInputError(f"{label} must be boolean")
    return bool(value)


def _validate_self_fingerprint(
    payload: Mapping[str, Any],
    *,
    fingerprint_field: str,
) -> dict[str, Any]:
    result = dict(payload)
    supplied = _require_sha256(
        result.get(fingerprint_field),
        fingerprint_field,
    )
    unsigned = {key: value for key, value in result.items() if key != fingerprint_field}
    if _json_sha256(unsigned) != supplied:
        raise PipelineInputError(f"{fingerprint_field} mismatch")
    return result


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value or "")
    if _SHA256_RE.fullmatch(digest) is None:
        raise PipelineInputError(f"{label} must be lowercase 64-hex SHA256")
    return digest


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def build_parser(*, defaults: Mapping[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "apply"):
        command = subparsers.add_parser(name)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--primitive-catalog")
        source.add_argument("--train")
        command.add_argument("--val")
        command.add_argument("--grouping-json", default=DEFAULT_GROUPING)
        command.add_argument("--coverage-proxy-artifact")
        command.add_argument("--phase-schema", default=DEFAULT_PHASE_SCHEMA)
        command.add_argument("--residual-mask")
        command.add_argument("--with-residual", action="store_true")
        command.add_argument(
            "--readiness",
            dest="readiness_mode",
            choices=("formal", "bootstrap"),
            default="formal",
        )
        command.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
        command.add_argument("--target-skill-id", default="ChinaJump")
        command.add_argument("--env-prefix", default=DEFAULT_ENV_PREFIX)
        command.add_argument("--formal-config-name", default=DEFAULT_FORMAL_CONFIG)
        command.add_argument("--residual-config-name", default=DEFAULT_RESIDUAL_CONFIG)
        command.add_argument("--bootstrap-config-name", default=DEFAULT_BOOTSTRAP_CONFIG)
        command.add_argument(
            "--fit-mode",
            choices=("global", "regional", "both"),
            default="both",
        )
        command.add_argument("--ranks", nargs="+", type=int, default=list(range(1, 11)))
        command.add_argument(
            "--region-ranks-json",
            help="JSON object mapping region labels to candidate rank lists",
        )
        command.add_argument("--total-rank-budget", type=int)
        command.add_argument("--require-dynamic-coverage", action="store_true")
        command.add_argument("--max-mean-dynamic-gap", type=float, default=0.15)
        command.add_argument(
            "--max-key-phase-dynamic-gap",
            type=float,
            default=0.25,
        )
        command.add_argument(
            "--max-basis-condition-number",
            type=float,
            default=1.0e6,
        )
        command.add_argument(
            "--min-effective-rank-fraction",
            type=float,
            default=1.0,
        )
        command.add_argument("--expected-environment-fingerprint")
        command.add_argument("--expected-rollout-manifest-fingerprint")
        command.add_argument(
            "--dynamic-coverage-reports-json",
            help=("strict signal-kind/region/rank report inventory produced from the first-stage candidate inventory"),
        )
        command.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
        command.add_argument(
            "--normalization",
            choices=("channel_max", "channel_l2", "none"),
            default="channel_max",
        )
        command.add_argument("--near-zero-threshold", type=float, default=1e-8)
        command.add_argument("--phase-weights-json")
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--release", required=True)
    preflight.add_argument("--config-name")
    preflight.add_argument("--real-env-smoke", action="store_true")
    preflight.add_argument("--output")
    if defaults:
        for command in subparsers.choices.values():
            command.set_defaults(**dict(defaults))
    return parser


def _request_from_args(args: argparse.Namespace) -> PipelineRequest:
    return PipelineRequest(
        train=args.train,
        val=args.val,
        primitive_catalog=args.primitive_catalog,
        grouping_json=args.grouping_json,
        coverage_proxy_artifact=args.coverage_proxy_artifact,
        phase_schema=args.phase_schema,
        residual_mask=args.residual_mask,
        output_root=args.output_root,
        target_skill_id=args.target_skill_id,
        env_prefix=args.env_prefix,
        readiness_mode=args.readiness_mode,
        with_residual=bool(args.with_residual),
        formal_config_name=args.formal_config_name,
        residual_config_name=args.residual_config_name,
        bootstrap_config_name=args.bootstrap_config_name,
        fit_mode=args.fit_mode,
        ranks=tuple(args.ranks),
        region_ranks=_load_region_ranks_json(args.region_ranks_json),
        total_rank_budget=args.total_rank_budget,
        require_dynamic_coverage=bool(args.require_dynamic_coverage),
        max_mean_dynamic_gap=float(args.max_mean_dynamic_gap),
        max_key_phase_dynamic_gap=float(args.max_key_phase_dynamic_gap),
        max_basis_condition_number=float(args.max_basis_condition_number),
        min_effective_rank_fraction=float(args.min_effective_rank_fraction),
        expected_environment_fingerprint=args.expected_environment_fingerprint,
        expected_rollout_manifest_fingerprint=(args.expected_rollout_manifest_fingerprint),
        dynamic_coverage_reports=_load_dynamic_coverage_reports_json(args.dynamic_coverage_reports_json),
        seeds=tuple(args.seeds),
        normalization=args.normalization,
        near_zero_threshold=float(args.near_zero_threshold),
        phase_weights_json=args.phase_weights_json,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    defaults: Mapping[str, Any] | None = None,
) -> int:
    args = build_parser(defaults=defaults).parse_args(argv)
    if args.command == "plan":
        result = plan_stage1_pipeline(_request_from_args(args))
    elif args.command == "apply":
        result = apply_stage1_pipeline(_request_from_args(args))
    else:
        result = preflight_stage1_release(
            args.release,
            config_name=args.config_name,
            real_env_smoke=bool(args.real_env_smoke),
        )
        if args.output:
            _atomic_write_json(Path(args.output), result)
    print(
        json.dumps(
            _jsonable(result),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
