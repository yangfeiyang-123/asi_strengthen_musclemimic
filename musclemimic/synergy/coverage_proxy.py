"""Content-addressed ChinaJump target-excitation coverage proxies.

This module deliberately does not infer muscle excitation from kinematics.
It accepts only raw controls that were applied by a full-dimensional teacher
or a full-dimensional trajectory optimizer, recomputes unit excitation from
the exact ordered actuator control ranges, and seals the result together with
source/QC provenance.  Primitive rollouts and early-synergy policies are
rejected because either would make the target coverage check circular.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.distill.physical import (
    MUSCLE_EXCITATION_FORMULA,
    MUSCLE_EXCITATION_ROUNDOFF_POLICY,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
    UNIT_INTERVAL_TOLERANCE,
    physical_ctrl_to_effective_muscle_excitation,
    validate_muscle_channel_contract,
    validate_ordered_ctrlrange,
    validate_unit_muscle_ctrlrange,
    validate_unit_muscle_excitation,
)
from musclemimic.distill.provenance import checkpoint_content_fingerprint
from musclemimic.synergy.oracle_coverage import (
    canonicalize_static_proxy_phase_schema,
    load_static_proxy_phase_schema,
    proxy_content_fingerprint,
)
from musclemimic.synergy.schema import EXCITATION_SIGNAL_KIND, ctrlrange_schema_hash

TARGET_CONTROL_SOURCE_SCHEMA_VERSION = "chinajump_target_control_source_v2"
TARGET_CONTROL_QC_SCHEMA_VERSION = "chinajump_target_control_qc_v2"
COVERAGE_PROXY_MANIFEST_SCHEMA_VERSION = "chinajump_excitation_proxy_manifest_v2"
COVERAGE_PROXY_ARTIFACT_KIND = "chinajump_target_physical_excitation_proxy"
COVERAGE_PROXY_FILENAME = "static_excitation_proxy.npz"
COVERAGE_PROXY_MANIFEST_FILENAME = "proxy_manifest.json"
COVERAGE_PROXY_SOURCE_MANIFEST_FILENAME = "source_manifest.json"
COVERAGE_PROXY_SOURCE_QC_FILENAME = "source_qc.json"

_SOURCE_KINDS = frozenset({"full_action_teacher", "trajectory_optimizer"})
_RAW_CONTROL_FIELDS = ("teacher_ctrl_physical", "applied_ctrl")
_CTRLRANGE_FIELDS = ("actuator_ctrlrange", "ctrlrange")
_IDENTITY_FIELDS = ("motion_uid", "rollout_uid", "subtraj_step_no", "frame_index")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "source_kind",
        "target_skill_id",
        "primitive_source",
        "contains_target_skill_controls",
        "action_space_kind",
        "early_synergy_action_representation",
        "action_dim",
        "source_npz_sha256",
        "model_hash",
        "actuator_schema_hash",
        "ctrlrange_schema_hash",
        "reference_fingerprint",
        "phase_annotation_fingerprint",
        "checkpoint_fingerprint",
        "optimizer_fingerprint",
        "control_artifact_audit",
        "manifest_fingerprint",
    }
)
_QC_FIELDS = frozenset(
    {
        "schema_version",
        "source_manifest_fingerprint",
        "target_skill_id",
        "passed",
        "tracking_qc_passed",
        "forward_replay_verified",
        "complete_trajectory_coverage",
        "early_termination_rate",
        "frame_coverage",
        "episode_success_rate",
        "excitation_upper_saturation_fraction",
        "observed_phase_ids",
        "per_phase_sample_counts",
        "model_hash",
        "actuator_schema_hash",
        "ctrlrange_schema_hash",
        "reference_fingerprint",
        "phase_annotation_fingerprint",
        "thresholds",
        "checks",
        "qc_fingerprint",
    }
)
_QC_THRESHOLD_FIELDS = frozenset(
    {
        "max_early_termination_rate",
        "min_frame_coverage",
        "min_episode_success_rate",
        "max_excitation_upper_saturation_fraction",
    }
)
_QC_CHECK_FIELDS = frozenset(
    {
        "tracking_qc_passed",
        "forward_replay_verified",
        "complete_trajectory_coverage",
        "early_termination_rate",
        "frame_coverage",
        "episode_success_rate",
        "excitation_upper_saturation_fraction",
    }
)
_CONTENT_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "supplied_path",
        "resolved_path",
        "sha256",
        "num_files",
        "num_bytes",
        "files",
    }
)
_CONTENT_AUDIT_FILE_FIELDS = frozenset({"path", "sha256", "num_bytes"})
_PROXY_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "target_skill_id",
        "signal_kind",
        "source_binding",
        "physical_binding",
        "phase_binding",
        "selection",
        "proxy_binding",
        "manifest_fingerprint",
    }
)
_SOURCE_BINDING_FIELDS = frozenset(
    {
        "source_manifest_schema_version",
        "source_manifest_fingerprint",
        "source_qc_schema_version",
        "source_qc_fingerprint",
        "source_kind",
        "source_npz_sha256",
        "reference_fingerprint",
        "phase_annotation_fingerprint",
        "checkpoint_fingerprint",
        "optimizer_fingerprint",
        "source_manifest_filename",
        "source_manifest_file_sha256",
        "source_qc_filename",
        "source_qc_file_sha256",
    }
)
_PHYSICAL_BINDING_FIELDS = frozenset(
    {
        "action_dim",
        "model_hash",
        "actuator_names",
        "actuator_schema_hash",
        "actuator_ctrlrange",
        "ctrlrange_schema_hash",
        "physical_signal_schema_version",
        "muscle_channel_contract",
        "transform",
    }
)
_PHASE_BINDING_FIELDS = frozenset(
    {
        "phase_schema",
        "phase_schema_fingerprint",
        "required_phase_ids",
        "observed_phase_ids",
        "per_phase_sample_counts",
    }
)
_SELECTION_FIELDS = frozenset({"kind", "min_phase_samples", "thresholds"})
_SELECTION_THRESHOLD_FIELDS = frozenset(
    {
        "max_early_termination_rate",
        "min_frame_coverage",
        "min_episode_success_rate",
        "max_excitation_upper_saturation_fraction",
    }
)
_PROXY_BINDING_FIELDS = frozenset(
    {
        "filename",
        "file_sha256",
        "content_fingerprint",
        "shape",
        "dtype",
        "identity_fields",
        "identity_fingerprint",
    }
)


@dataclass(frozen=True)
class CoverageProxyArtifact:
    """A fully revalidated coverage proxy and its portable identities."""

    manifest: dict[str, Any]
    manifest_path: Path
    npz_path: Path
    source_manifest_path: Path
    source_qc_path: Path
    manifest_fingerprint: str
    content_fingerprint: str
    source_manifest_fingerprint: str
    source_qc_fingerprint: str
    source_kind: str

    @property
    def required_phase_ids(self) -> tuple[int, ...]:
        """The exact producer-side phase inventory required for this proxy."""

        return tuple(int(value) for value in self.manifest["phase_binding"]["required_phase_ids"])

    @property
    def min_phase_samples(self) -> int:
        """The sealed minimum sample count for every required phase."""

        return int(self.manifest["selection"]["min_phase_samples"])

    @property
    def per_phase_sample_counts(self) -> dict[int, int]:
        """Return the revalidated phase counts in integer-key form."""

        return {
            int(key): int(value) for key, value in self.manifest["phase_binding"]["per_phase_sample_counts"].items()
        }

    @property
    def oracle_binding(self) -> dict[str, Any]:
        phase = self.manifest["phase_binding"]
        return {
            "producer_manifest_schema_version": self.manifest["schema_version"],
            "producer_manifest_fingerprint": self.manifest_fingerprint,
            "producer_artifact_kind": self.manifest["artifact_kind"],
            "source_kind": self.source_kind,
            "source_manifest_fingerprint": self.source_manifest_fingerprint,
            "source_qc_fingerprint": self.source_qc_fingerprint,
            "proxy_content_fingerprint": self.content_fingerprint,
            "phase_schema_fingerprint": phase["phase_schema_fingerprint"],
            "required_phase_ids": list(self.required_phase_ids),
            "min_phase_samples": self.min_phase_samples,
            "per_phase_sample_counts": {str(key): value for key, value in self.per_phase_sample_counts.items()},
        }


def build_target_control_source_manifest(
    input_npz: str | Path,
    *,
    source_kind: str,
    target_skill_id: str,
    action_dim: int,
    model_hash: str,
    actuator_schema_fingerprint: str,
    ctrlrange_schema_fingerprint: str,
    reference_fingerprint: str,
    phase_annotation_fingerprint: str,
    checkpoint_artifact_path: str | Path | None = None,
    optimizer_artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build provenance by hashing the live full-action control artifact.

    Fingerprints are intentionally not accepted as caller input.  A teacher
    checkpoint or optimizer artifact must exist while this function runs, and
    its complete file inventory is sealed into the source manifest.
    """

    kind = str(source_kind)
    if kind == "full_action_teacher":
        if checkpoint_artifact_path is None or optimizer_artifact_path is not None:
            raise ValueError("full_action_teacher requires checkpoint_artifact_path and no optimizer_artifact_path")
        checkpoint_path = Path(checkpoint_artifact_path).expanduser()
        control_audit = (
            _full_file_content_fingerprint(checkpoint_path)
            if checkpoint_path.is_file()
            else checkpoint_content_fingerprint(checkpoint_path)
        )
    elif kind == "trajectory_optimizer":
        if optimizer_artifact_path is None or checkpoint_artifact_path is not None:
            raise ValueError("trajectory_optimizer requires optimizer_artifact_path and no checkpoint_artifact_path")
        control_audit = _full_file_content_fingerprint(optimizer_artifact_path)
    else:
        raise ValueError(
            "coverage proxy source_kind must be full_action_teacher or trajectory_optimizer; "
            "primitive sources are forbidden"
        )
    control_audit = _validate_content_audit(
        control_audit,
        source_kind=kind,
        verify_live_artifact=True,
    )
    control_fingerprint = control_audit["sha256"]
    source_action_dim = _strict_int(action_dim, "action_dim")
    if source_action_dim <= 0:
        raise ValueError("target-control action_dim must be positive")
    payload = {
        "schema_version": TARGET_CONTROL_SOURCE_SCHEMA_VERSION,
        "source_kind": kind,
        "target_skill_id": _nonempty_string(target_skill_id, "target_skill_id"),
        "primitive_source": False,
        "contains_target_skill_controls": True,
        "action_space_kind": "full_muscle_control",
        "early_synergy_action_representation": False,
        "action_dim": source_action_dim,
        "source_npz_sha256": _file_sha256(_existing_file(input_npz, "input control NPZ")),
        "model_hash": _require_sha256(model_hash, "model_hash"),
        "actuator_schema_hash": _require_sha256(
            actuator_schema_fingerprint,
            "actuator_schema_fingerprint",
        ),
        "ctrlrange_schema_hash": _require_sha256(
            ctrlrange_schema_fingerprint,
            "ctrlrange_schema_fingerprint",
        ),
        "reference_fingerprint": _require_sha256(
            reference_fingerprint,
            "reference_fingerprint",
        ),
        "phase_annotation_fingerprint": _require_sha256(
            phase_annotation_fingerprint,
            "phase_annotation_fingerprint",
        ),
        "checkpoint_fingerprint": control_fingerprint if kind == "full_action_teacher" else None,
        "optimizer_fingerprint": control_fingerprint if kind == "trajectory_optimizer" else None,
        "control_artifact_audit": control_audit,
    }
    payload["manifest_fingerprint"] = _json_fingerprint(payload, excluded=("manifest_fingerprint",))
    return validate_target_control_source_manifest(payload)


def write_target_control_source_manifest(
    path: str | Path,
    input_npz: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_target_control_source_manifest(input_npz, **kwargs)
    _atomic_write_json(Path(path), payload)
    return load_target_control_source_manifest(path)


def load_target_control_source_manifest(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    source = _existing_file(path, "target-control source manifest")
    payload = load_json_strict(source)
    return validate_target_control_source_manifest(
        payload,
        expected_fingerprint=expected_fingerprint,
    )


def validate_target_control_source_manifest(
    payload: Mapping[str, Any],
    *,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    result = _strict_mapping(payload, _SOURCE_FIELDS, "target-control source manifest")
    if result["schema_version"] != TARGET_CONTROL_SOURCE_SCHEMA_VERSION:
        raise ValueError("unsupported target-control source manifest schema")
    kind = result["source_kind"]
    if kind not in _SOURCE_KINDS:
        raise ValueError(
            "coverage proxy source_kind must be full_action_teacher or trajectory_optimizer; "
            "primitive sources are forbidden"
        )
    if result["primitive_source"] is not False or result["contains_target_skill_controls"] is not True:
        raise ValueError("coverage proxy source must contain target-skill controls and cannot be primitive-only")
    if result["action_space_kind"] != "full_muscle_control":
        raise ValueError("coverage proxy requires a full-dimensional muscle-control source")
    if result["early_synergy_action_representation"] is not False:
        raise ValueError("early-synergy teacher coverage is circular and forbidden")
    _nonempty_string(result["target_skill_id"], "target_skill_id")
    if _strict_int(result["action_dim"], "action_dim") <= 0:
        raise ValueError("target-control action_dim must be positive")
    for field in (
        "source_npz_sha256",
        "model_hash",
        "actuator_schema_hash",
        "ctrlrange_schema_hash",
        "reference_fingerprint",
        "phase_annotation_fingerprint",
    ):
        _require_sha256(result[field], field)
    checkpoint = _optional_sha256(result["checkpoint_fingerprint"], "checkpoint_fingerprint")
    optimizer = _optional_sha256(result["optimizer_fingerprint"], "optimizer_fingerprint")
    if kind == "full_action_teacher" and (checkpoint is None or optimizer is not None):
        raise ValueError("full_action_teacher requires checkpoint_fingerprint and no optimizer_fingerprint")
    if kind == "trajectory_optimizer" and (optimizer is None or checkpoint is not None):
        raise ValueError("trajectory_optimizer requires optimizer_fingerprint and no checkpoint_fingerprint")
    control_audit = _validate_content_audit(
        result["control_artifact_audit"],
        source_kind=kind,
        verify_live_artifact=True,
    )
    result["control_artifact_audit"] = control_audit
    expected_control_fingerprint = checkpoint if kind == "full_action_teacher" else optimizer
    if control_audit["sha256"] != expected_control_fingerprint:
        raise ValueError("target-control artifact audit differs from its derived source fingerprint")
    supplied = _require_sha256(result["manifest_fingerprint"], "manifest_fingerprint")
    expected = _json_fingerprint(result, excluded=("manifest_fingerprint",))
    if supplied != expected:
        raise ValueError("target-control source manifest_fingerprint mismatch")
    if expected_fingerprint is not None and supplied != _require_sha256(
        expected_fingerprint,
        "expected source manifest fingerprint",
    ):
        raise ValueError("target-control source manifest differs from expected fingerprint")
    return result


def build_target_control_qc(
    source_manifest: Mapping[str, Any],
    *,
    phase_id: np.ndarray,
    excitation_upper_saturation_fraction: float,
    tracking_qc_passed: bool,
    forward_replay_verified: bool,
    complete_trajectory_coverage: bool,
    early_termination_rate: float,
    frame_coverage: float,
    episode_success_rate: float,
    max_early_termination_rate: float,
    min_frame_coverage: float,
    min_episode_success_rate: float,
    max_excitation_upper_saturation_fraction: float,
) -> dict[str, Any]:
    """Build QC from explicit measurements and derive ``passed`` fail-closed."""

    source = validate_target_control_source_manifest(source_manifest)
    phases = _validate_phase_id(phase_id)
    counts = _phase_counts(phases)
    tracking = _strict_bool(tracking_qc_passed, "tracking_qc_passed")
    replay = _strict_bool(forward_replay_verified, "forward_replay_verified")
    complete = _strict_bool(complete_trajectory_coverage, "complete_trajectory_coverage")
    early_rate = _unit_fraction(early_termination_rate, "early_termination_rate")
    coverage = _unit_fraction(frame_coverage, "frame_coverage")
    success = _unit_fraction(episode_success_rate, "episode_success_rate")
    saturation = _unit_fraction(
        excitation_upper_saturation_fraction,
        "excitation_upper_saturation_fraction",
    )
    thresholds = {
        "max_early_termination_rate": _unit_fraction(
            max_early_termination_rate,
            "max_early_termination_rate",
        ),
        "min_frame_coverage": _unit_fraction(min_frame_coverage, "min_frame_coverage"),
        "min_episode_success_rate": _unit_fraction(
            min_episode_success_rate,
            "min_episode_success_rate",
        ),
        "max_excitation_upper_saturation_fraction": _unit_fraction(
            max_excitation_upper_saturation_fraction,
            "max_excitation_upper_saturation_fraction",
        ),
    }
    checks = _target_control_qc_checks(
        tracking_qc_passed=tracking,
        forward_replay_verified=replay,
        complete_trajectory_coverage=complete,
        early_termination_rate=early_rate,
        frame_coverage=coverage,
        episode_success_rate=success,
        excitation_upper_saturation_fraction=saturation,
        thresholds=thresholds,
    )
    payload = {
        "schema_version": TARGET_CONTROL_QC_SCHEMA_VERSION,
        "source_manifest_fingerprint": source["manifest_fingerprint"],
        "target_skill_id": source["target_skill_id"],
        "passed": bool(all(checks.values())),
        "tracking_qc_passed": tracking,
        "forward_replay_verified": replay,
        "complete_trajectory_coverage": complete,
        "early_termination_rate": early_rate,
        "frame_coverage": coverage,
        "episode_success_rate": success,
        "excitation_upper_saturation_fraction": saturation,
        "observed_phase_ids": sorted(int(value) for value in counts),
        "per_phase_sample_counts": {str(key): value for key, value in counts.items()},
        "model_hash": source["model_hash"],
        "actuator_schema_hash": source["actuator_schema_hash"],
        "ctrlrange_schema_hash": source["ctrlrange_schema_hash"],
        "reference_fingerprint": source["reference_fingerprint"],
        "phase_annotation_fingerprint": source["phase_annotation_fingerprint"],
        "thresholds": thresholds,
        "checks": checks,
    }
    payload["qc_fingerprint"] = _json_fingerprint(payload, excluded=("qc_fingerprint",))
    return validate_target_control_qc(payload, source_manifest=source)


def write_target_control_qc(
    path: str | Path,
    source_manifest: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_target_control_qc(source_manifest, **kwargs)
    _atomic_write_json(Path(path), payload)
    return load_target_control_qc(path, source_manifest=source_manifest)


def load_target_control_qc(
    path: str | Path,
    *,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    source = _existing_file(path, "target-control QC manifest")
    payload = load_json_strict(source)
    return validate_target_control_qc(payload, source_manifest=source_manifest)


def validate_target_control_qc(
    payload: Mapping[str, Any],
    *,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    source = validate_target_control_source_manifest(source_manifest)
    result = _strict_mapping(payload, _QC_FIELDS, "target-control QC")
    if result["schema_version"] != TARGET_CONTROL_QC_SCHEMA_VERSION:
        raise ValueError("unsupported target-control QC schema")
    if result["source_manifest_fingerprint"] != source["manifest_fingerprint"]:
        raise ValueError("target-control QC source manifest fingerprint drift")
    if result["target_skill_id"] != source["target_skill_id"]:
        raise ValueError("target-control QC target skill drift")
    for field in (
        "model_hash",
        "actuator_schema_hash",
        "ctrlrange_schema_hash",
        "reference_fingerprint",
        "phase_annotation_fingerprint",
    ):
        if _require_sha256(result[field], f"QC {field}") != source[field]:
            raise ValueError(f"target-control QC {field} drift")
    for field in ("passed", "tracking_qc_passed", "forward_replay_verified", "complete_trajectory_coverage"):
        _strict_bool(result[field], f"target-control QC {field}")
    for field in (
        "early_termination_rate",
        "frame_coverage",
        "episode_success_rate",
        "excitation_upper_saturation_fraction",
    ):
        _unit_fraction(result[field], f"target-control QC {field}")
    observed = _int_list(result["observed_phase_ids"], "observed_phase_ids")
    if observed != sorted(set(observed)) or any(value < 0 for value in observed):
        raise ValueError("target-control QC observed_phase_ids must be sorted unique non-negative integers")
    counts = _phase_count_mapping(result["per_phase_sample_counts"])
    if observed != sorted(counts):
        raise ValueError("target-control QC phase inventory differs from per-phase counts")
    thresholds = _strict_mapping(result["thresholds"], _QC_THRESHOLD_FIELDS, "target-control QC thresholds")
    canonical_thresholds = {field: _unit_fraction(thresholds[field], field) for field in _QC_THRESHOLD_FIELDS}
    if dict(thresholds) != canonical_thresholds:
        raise ValueError("target-control QC thresholds are not canonical finite fractions")
    checks = _strict_mapping(result["checks"], _QC_CHECK_FIELDS, "target-control QC checks")
    if any(type(value) is not bool for value in checks.values()):
        raise ValueError("target-control QC checks must be booleans")
    recomputed_checks = _target_control_qc_checks(
        tracking_qc_passed=result["tracking_qc_passed"],
        forward_replay_verified=result["forward_replay_verified"],
        complete_trajectory_coverage=result["complete_trajectory_coverage"],
        early_termination_rate=result["early_termination_rate"],
        frame_coverage=result["frame_coverage"],
        episode_success_rate=result["episode_success_rate"],
        excitation_upper_saturation_fraction=result["excitation_upper_saturation_fraction"],
        thresholds=canonical_thresholds,
    )
    if checks != recomputed_checks:
        raise ValueError("target-control QC checks are stale or inconsistent with metrics")
    if result["passed"] is not bool(all(recomputed_checks.values())):
        raise ValueError("target-control QC passed flag is stale or inconsistent with checks")
    supplied = _require_sha256(result["qc_fingerprint"], "qc_fingerprint")
    expected = _json_fingerprint(result, excluded=("qc_fingerprint",))
    if supplied != expected:
        raise ValueError("target-control QC fingerprint mismatch")
    return result


def build_coverage_proxy(
    input_npz: str | Path,
    *,
    source_manifest_path: str | Path,
    source_qc_path: str | Path,
    phase_schema_path: str | Path,
    output_dir: str | Path,
    expected_target_skill_id: str = "ChinaJump",
    expected_action_dim: int = 354,
    required_phase_ids: Sequence[int] = (1, 2, 3, 4),
    min_phase_samples: int = 2,
    max_early_termination_rate: float = 0.05,
    min_frame_coverage: float = 0.95,
    min_episode_success_rate: float = 0.95,
    max_excitation_upper_saturation_fraction: float = 0.05,
    expected_model_hash: str | None = None,
    expected_reference_fingerprint: str | None = None,
    expected_control_source_fingerprint: str | None = None,
) -> CoverageProxyArtifact:
    """Validate and seal a target-control NPZ into a formal coverage proxy."""

    input_path = _existing_file(input_npz, "input control NPZ")
    source = load_target_control_source_manifest(source_manifest_path)
    qc = load_target_control_qc(source_qc_path, source_manifest=source)
    phase_schema = load_static_proxy_phase_schema(phase_schema_path)
    target_skill_id = _nonempty_string(expected_target_skill_id, "expected_target_skill_id")
    if source["target_skill_id"] != target_skill_id or phase_schema["target_skill_id"] != target_skill_id:
        raise ValueError("source/phase schema target skill differs from expected target skill")
    action_dim = _strict_int(expected_action_dim, "expected_action_dim")
    if action_dim <= 0 or source["action_dim"] != action_dim:
        raise ValueError("target-control source action dimension drift")
    if source["source_npz_sha256"] != _file_sha256(input_path):
        raise ValueError("input control NPZ differs from source manifest content hash")
    if expected_model_hash is not None and source["model_hash"] != _require_sha256(
        expected_model_hash,
        "expected_model_hash",
    ):
        raise ValueError("target-control model hash drift")
    if expected_reference_fingerprint is not None and source["reference_fingerprint"] != _require_sha256(
        expected_reference_fingerprint,
        "expected_reference_fingerprint",
    ):
        raise ValueError("target-control reference fingerprint drift")
    if expected_control_source_fingerprint is not None:
        actual_control_source = source["checkpoint_fingerprint"] or source["optimizer_fingerprint"]
        if actual_control_source != _require_sha256(
            expected_control_source_fingerprint,
            "expected_control_source_fingerprint",
        ):
            raise ValueError("target-control checkpoint/optimizer fingerprint drift")

    arrays = _load_control_arrays(input_path, expected_action_dim=action_dim)
    expected_raw_field = "teacher_ctrl_physical" if source["source_kind"] == "full_action_teacher" else "applied_ctrl"
    if arrays["raw_field"] != expected_raw_field:
        raise ValueError(
            f"target-control raw field {arrays['raw_field']!r} is inconsistent with source_kind {source['source_kind']!r}"
        )
    names = arrays["actuator_names"]
    ctrlrange = validate_unit_muscle_ctrlrange(
        names,
        arrays["actuator_ctrlrange"],
    )
    raw_ctrl = arrays["raw_ctrl"]
    phase_id = arrays["phase_id"]
    channel_contract = validate_muscle_channel_contract(
        arrays["muscle_channel_contract"],
        expected_names=names,
    )
    excitation = validate_unit_muscle_excitation(
        physical_ctrl_to_effective_muscle_excitation(
            raw_ctrl,
            channel_contract=channel_contract,
        )
    )
    if arrays["declared_excitation"] is not None:
        declared = validate_unit_muscle_excitation(arrays["declared_excitation"])
        if declared.shape != excitation.shape or not np.allclose(declared, excitation, rtol=1e-6, atol=1e-6):
            raise ValueError("declared physical_excitation differs from clip(raw data.ctrl,0,1)")

    action_schema = actuator_schema_hash(names)
    control_schema = ctrlrange_schema_hash(names, ctrlrange)
    if action_schema != source["actuator_schema_hash"]:
        raise ValueError("input actuator names/order drift from source manifest")
    if control_schema != source["ctrlrange_schema_hash"]:
        raise ValueError("input ordered ctrlrange drift from source manifest")
    _validate_phase_semantics(phase_id, phase_schema)
    required = _canonical_required_phases(required_phase_ids, phase_schema)
    counts = _phase_counts(phase_id)
    minimum_samples = _strict_int(min_phase_samples, "min_phase_samples")
    if minimum_samples <= 0:
        raise ValueError("min_phase_samples must be positive")
    missing_or_short = {phase: counts.get(phase, 0) for phase in required if counts.get(phase, 0) < minimum_samples}
    if missing_or_short:
        raise ValueError(f"coverage proxy required phases are missing or undersampled: {missing_or_short}")
    _validate_qc_against_arrays(
        qc,
        source=source,
        phase_counts=counts,
        excitation=excitation,
        max_early_termination_rate=max_early_termination_rate,
        min_frame_coverage=min_frame_coverage,
        min_episode_success_rate=min_episode_success_rate,
        max_excitation_upper_saturation_fraction=max_excitation_upper_saturation_fraction,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    embedded_source_path = output / COVERAGE_PROXY_SOURCE_MANIFEST_FILENAME
    embedded_qc_path = output / COVERAGE_PROXY_SOURCE_QC_FILENAME
    _atomic_write_json(embedded_source_path, source)
    embedded_source = load_target_control_source_manifest(
        embedded_source_path,
        expected_fingerprint=source["manifest_fingerprint"],
    )
    _atomic_write_json(embedded_qc_path, qc)
    embedded_qc = load_target_control_qc(
        embedded_qc_path,
        source_manifest=embedded_source,
    )
    npz_path = output / COVERAGE_PROXY_FILENAME
    identity = arrays["identity"]
    npz_payload: dict[str, np.ndarray] = {
        "physical_excitation": np.asarray(excitation, dtype=np.float32),
        arrays["raw_field"]: np.asarray(raw_ctrl, dtype=np.float32),
        "phase_id": np.asarray(phase_id, dtype=np.int32),
        "actuator_names": np.asarray(names),
        "actuator_ctrlrange": np.asarray(ctrlrange, dtype=np.float64),
        "source_row_index": np.arange(excitation.shape[0], dtype=np.int64),
    }
    npz_payload.update(identity)
    _atomic_write_npz(npz_path, npz_payload)
    content_fingerprint = proxy_content_fingerprint(
        excitation,
        muscle_names=names,
        phase_id=phase_id,
        phase_schema=phase_schema,
    )
    thresholds = {
        "max_early_termination_rate": _unit_fraction(
            max_early_termination_rate,
            "max_early_termination_rate",
        ),
        "min_frame_coverage": _unit_fraction(min_frame_coverage, "min_frame_coverage"),
        "min_episode_success_rate": _unit_fraction(
            min_episode_success_rate,
            "min_episode_success_rate",
        ),
        "max_excitation_upper_saturation_fraction": _unit_fraction(
            max_excitation_upper_saturation_fraction,
            "max_excitation_upper_saturation_fraction",
        ),
    }
    manifest = {
        "schema_version": COVERAGE_PROXY_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": COVERAGE_PROXY_ARTIFACT_KIND,
        "target_skill_id": target_skill_id,
        "signal_kind": EXCITATION_SIGNAL_KIND,
        "source_binding": {
            "source_manifest_schema_version": source["schema_version"],
            "source_manifest_fingerprint": source["manifest_fingerprint"],
            "source_qc_schema_version": qc["schema_version"],
            "source_qc_fingerprint": embedded_qc["qc_fingerprint"],
            "source_kind": source["source_kind"],
            "source_npz_sha256": source["source_npz_sha256"],
            "reference_fingerprint": source["reference_fingerprint"],
            "phase_annotation_fingerprint": source["phase_annotation_fingerprint"],
            "checkpoint_fingerprint": source["checkpoint_fingerprint"],
            "optimizer_fingerprint": source["optimizer_fingerprint"],
            "source_manifest_filename": COVERAGE_PROXY_SOURCE_MANIFEST_FILENAME,
            "source_manifest_file_sha256": _file_sha256(embedded_source_path),
            "source_qc_filename": COVERAGE_PROXY_SOURCE_QC_FILENAME,
            "source_qc_file_sha256": _file_sha256(embedded_qc_path),
        },
        "physical_binding": {
            "action_dim": action_dim,
            "model_hash": source["model_hash"],
            "actuator_names": list(names),
            "actuator_schema_hash": action_schema,
            "actuator_ctrlrange": ctrlrange.tolist(),
            "ctrlrange_schema_hash": control_schema,
            "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
            "muscle_channel_contract": channel_contract.to_metadata(),
            "transform": {
                "kind": UNIT_EXCITATION_TRANSFORM,
                "raw_signal_kind": arrays["raw_field"],
                "formula": MUSCLE_EXCITATION_FORMULA,
                "roundoff_policy": MUSCLE_EXCITATION_ROUNDOFF_POLICY,
            },
        },
        "phase_binding": {
            "phase_schema": phase_schema,
            "phase_schema_fingerprint": phase_schema["phase_schema_fingerprint"],
            "required_phase_ids": list(required),
            "observed_phase_ids": sorted(counts),
            "per_phase_sample_counts": {str(key): value for key, value in counts.items()},
        },
        "selection": {
            "kind": "all_qc_passed_target_control_frames",
            "min_phase_samples": minimum_samples,
            "thresholds": thresholds,
        },
        "proxy_binding": {
            "filename": COVERAGE_PROXY_FILENAME,
            "file_sha256": _file_sha256(npz_path),
            "content_fingerprint": content_fingerprint,
            "shape": list(excitation.shape),
            "dtype": "float32",
            "identity_fields": sorted(identity),
            "identity_fingerprint": _identity_fingerprint(identity, excitation.shape[0]),
        },
    }
    manifest["manifest_fingerprint"] = _json_fingerprint(
        manifest,
        excluded=("manifest_fingerprint",),
    )
    manifest_path = output / COVERAGE_PROXY_MANIFEST_FILENAME
    _atomic_write_json(manifest_path, manifest)
    return load_coverage_proxy_artifact(manifest_path)


def load_coverage_proxy_artifact(
    path: str | Path,
    *,
    expected_manifest_fingerprint: str | None = None,
    expected_content_fingerprint: str | None = None,
) -> CoverageProxyArtifact:
    """Load a manifest/directory/NPZ and revalidate both metadata and bytes."""

    manifest_path = _resolve_proxy_manifest_path(path)
    payload = load_json_strict(manifest_path)
    manifest = _validate_proxy_manifest(payload)
    supplied_manifest_fingerprint = manifest["manifest_fingerprint"]
    if expected_manifest_fingerprint is not None and supplied_manifest_fingerprint != _require_sha256(
        expected_manifest_fingerprint,
        "expected coverage proxy manifest fingerprint",
    ):
        raise ValueError("coverage proxy manifest differs from expected fingerprint")
    source_binding = manifest["source_binding"]
    embedded_source_path = (manifest_path.parent / source_binding["source_manifest_filename"]).resolve()
    embedded_qc_path = (manifest_path.parent / source_binding["source_qc_filename"]).resolve()
    if not embedded_source_path.is_file() or not embedded_qc_path.is_file():
        raise FileNotFoundError("coverage proxy embedded source manifest/QC is missing")
    if _file_sha256(embedded_source_path) != source_binding["source_manifest_file_sha256"]:
        raise ValueError("coverage proxy embedded source manifest content hash mismatch")
    if _file_sha256(embedded_qc_path) != source_binding["source_qc_file_sha256"]:
        raise ValueError("coverage proxy embedded source QC content hash mismatch")
    embedded_source = load_target_control_source_manifest(
        embedded_source_path,
        expected_fingerprint=source_binding["source_manifest_fingerprint"],
    )
    embedded_qc = load_target_control_qc(
        embedded_qc_path,
        source_manifest=embedded_source,
    )
    _validate_embedded_source_binding(
        manifest,
        source=embedded_source,
        qc=embedded_qc,
    )
    proxy_binding = manifest["proxy_binding"]
    npz_path = (manifest_path.parent / proxy_binding["filename"]).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"coverage proxy NPZ does not exist: {npz_path}")
    if _file_sha256(npz_path) != proxy_binding["file_sha256"]:
        raise ValueError("coverage proxy NPZ content hash mismatch")
    physical = manifest["physical_binding"]
    phase_binding = manifest["phase_binding"]
    expected_raw_field = physical["transform"]["raw_signal_kind"]
    with np.load(npz_path, allow_pickle=False) as data:
        required = {
            "physical_excitation",
            expected_raw_field,
            "phase_id",
            "actuator_names",
            "actuator_ctrlrange",
            "source_row_index",
        }
        missing = sorted(required - set(data.files))
        unknown = sorted(set(data.files) - required - set(_IDENTITY_FIELDS))
        if missing or unknown:
            raise ValueError(f"coverage proxy NPZ fields differ from contract: missing={missing} unknown={unknown}")
        excitation = validate_unit_muscle_excitation(data["physical_excitation"])
        phase_id = _validate_phase_id(data["phase_id"], sample_count=excitation.shape[0])
        names = _actuator_names(data["actuator_names"], expected_width=excitation.shape[1])
        ctrlrange = validate_unit_muscle_ctrlrange(names, data["actuator_ctrlrange"])
        raw_ctrl = np.asarray(data[expected_raw_field], dtype=np.float64)
        channel_contract = validate_muscle_channel_contract(
            physical["muscle_channel_contract"],
            expected_names=names,
        )
        recomputed = physical_ctrl_to_effective_muscle_excitation(
            raw_ctrl,
            channel_contract=channel_contract,
        )
        if raw_ctrl.shape != excitation.shape or not np.allclose(
            excitation,
            recomputed,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError("coverage proxy excitation differs from retained raw data.ctrl")
        rows = np.asarray(data["source_row_index"])
        if rows.shape != (excitation.shape[0],) or not np.array_equal(
            rows,
            np.arange(excitation.shape[0], dtype=rows.dtype),
        ):
            raise ValueError("coverage proxy source_row_index is not canonical")
        identity = {field: np.asarray(data[field]) for field in _IDENTITY_FIELDS if field in data.files}
    if list(excitation.shape) != proxy_binding["shape"] or str(excitation.dtype) != proxy_binding["dtype"]:
        raise ValueError("coverage proxy array shape/dtype differs from manifest")
    if list(names) != physical["actuator_names"]:
        raise ValueError("coverage proxy actuator order differs from manifest")
    if not np.array_equal(ctrlrange, np.asarray(physical["actuator_ctrlrange"], dtype=np.float64)):
        raise ValueError("coverage proxy ctrlrange differs from manifest")
    _validate_phase_semantics(phase_id, phase_binding["phase_schema"])
    actual_counts = _phase_counts(phase_id)
    if {str(key): value for key, value in actual_counts.items()} != phase_binding["per_phase_sample_counts"]:
        raise ValueError("coverage proxy phase counts differ from manifest")
    content_fingerprint = proxy_content_fingerprint(
        excitation,
        muscle_names=names,
        phase_id=phase_id,
        phase_schema=phase_binding["phase_schema"],
    )
    if content_fingerprint != proxy_binding["content_fingerprint"]:
        raise ValueError("coverage proxy content fingerprint mismatch")
    if expected_content_fingerprint is not None and content_fingerprint != _require_sha256(
        expected_content_fingerprint,
        "expected coverage proxy content fingerprint",
    ):
        raise ValueError("coverage proxy content differs from expected fingerprint")
    if (
        sorted(identity) != proxy_binding["identity_fields"]
        or _identity_fingerprint(
            identity,
            excitation.shape[0],
        )
        != proxy_binding["identity_fingerprint"]
    ):
        raise ValueError("coverage proxy identity arrays differ from manifest")
    selection_thresholds = manifest["selection"]["thresholds"]
    _validate_qc_against_arrays(
        embedded_qc,
        source=embedded_source,
        phase_counts=actual_counts,
        excitation=excitation,
        max_early_termination_rate=selection_thresholds["max_early_termination_rate"],
        min_frame_coverage=selection_thresholds["min_frame_coverage"],
        min_episode_success_rate=selection_thresholds["min_episode_success_rate"],
        max_excitation_upper_saturation_fraction=selection_thresholds["max_excitation_upper_saturation_fraction"],
    )
    return CoverageProxyArtifact(
        manifest=manifest,
        manifest_path=manifest_path,
        npz_path=npz_path,
        source_manifest_path=embedded_source_path,
        source_qc_path=embedded_qc_path,
        manifest_fingerprint=supplied_manifest_fingerprint,
        content_fingerprint=content_fingerprint,
        source_manifest_fingerprint=source_binding["source_manifest_fingerprint"],
        source_qc_fingerprint=source_binding["source_qc_fingerprint"],
        source_kind=source_binding["source_kind"],
    )


def _validate_proxy_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _strict_mapping(payload, _PROXY_FIELDS, "coverage proxy manifest")
    if result["schema_version"] != COVERAGE_PROXY_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported coverage proxy manifest schema")
    if result["artifact_kind"] != COVERAGE_PROXY_ARTIFACT_KIND:
        raise ValueError("coverage proxy artifact kind is unsupported")
    _nonempty_string(result["target_skill_id"], "coverage proxy target_skill_id")
    if result["signal_kind"] != EXCITATION_SIGNAL_KIND:
        raise ValueError("coverage proxy signal must be physical unit excitation")
    source = _strict_mapping(result["source_binding"], _SOURCE_BINDING_FIELDS, "source_binding")
    if source["source_manifest_schema_version"] != TARGET_CONTROL_SOURCE_SCHEMA_VERSION:
        raise ValueError("coverage proxy source manifest schema drift")
    if source["source_qc_schema_version"] != TARGET_CONTROL_QC_SCHEMA_VERSION:
        raise ValueError("coverage proxy source QC schema drift")
    if source["source_kind"] not in _SOURCE_KINDS:
        raise ValueError("coverage proxy source kind is not a full-action source")
    for field in (
        "source_manifest_fingerprint",
        "source_qc_fingerprint",
        "source_npz_sha256",
        "reference_fingerprint",
        "phase_annotation_fingerprint",
        "source_manifest_file_sha256",
        "source_qc_file_sha256",
    ):
        _require_sha256(source[field], f"coverage proxy {field}")
    if source["source_manifest_filename"] != COVERAGE_PROXY_SOURCE_MANIFEST_FILENAME:
        raise ValueError("coverage proxy embedded source manifest filename is not canonical")
    if source["source_qc_filename"] != COVERAGE_PROXY_SOURCE_QC_FILENAME:
        raise ValueError("coverage proxy embedded source QC filename is not canonical")
    checkpoint = _optional_sha256(source["checkpoint_fingerprint"], "checkpoint_fingerprint")
    optimizer = _optional_sha256(source["optimizer_fingerprint"], "optimizer_fingerprint")
    if source["source_kind"] == "full_action_teacher" and (checkpoint is None or optimizer is not None):
        raise ValueError("coverage proxy full-action source identity is inconsistent")
    if source["source_kind"] == "trajectory_optimizer" and (optimizer is None or checkpoint is not None):
        raise ValueError("coverage proxy optimizer source identity is inconsistent")

    physical = _strict_mapping(result["physical_binding"], _PHYSICAL_BINDING_FIELDS, "physical_binding")
    action_dim = _strict_int(physical["action_dim"], "physical action_dim")
    names = _actuator_names(physical["actuator_names"], expected_width=action_dim)
    ctrlrange = validate_unit_muscle_ctrlrange(names, physical["actuator_ctrlrange"])
    if _require_sha256(physical["actuator_schema_hash"], "actuator_schema_hash") != actuator_schema_hash(names):
        raise ValueError("coverage proxy actuator schema hash mismatch")
    if _require_sha256(physical["ctrlrange_schema_hash"], "ctrlrange_schema_hash") != ctrlrange_schema_hash(
        names,
        ctrlrange,
    ):
        raise ValueError("coverage proxy ctrlrange schema hash mismatch")
    _require_sha256(physical["model_hash"], "model_hash")
    if physical["physical_signal_schema_version"] != PHYSICAL_SIGNAL_SCHEMA_VERSION:
        raise ValueError("coverage proxy physical signal schema is legacy or unsupported")
    validate_muscle_channel_contract(
        physical["muscle_channel_contract"],
        expected_names=names,
    )
    transform = _strict_mapping(
        physical["transform"],
        frozenset({"kind", "raw_signal_kind", "formula", "roundoff_policy"}),
        "physical transform",
    )
    if (
        transform["kind"] != UNIT_EXCITATION_TRANSFORM
        or transform["raw_signal_kind"] not in _RAW_CONTROL_FIELDS
        or transform["formula"] != MUSCLE_EXCITATION_FORMULA
        or transform["roundoff_policy"] != MUSCLE_EXCITATION_ROUNDOFF_POLICY
    ):
        raise ValueError("coverage proxy physical transform contract is unsupported")
    expected_raw_field = "teacher_ctrl_physical" if source["source_kind"] == "full_action_teacher" else "applied_ctrl"
    if transform["raw_signal_kind"] != expected_raw_field:
        raise ValueError("coverage proxy raw signal kind differs from full-action source kind")

    phase = _strict_mapping(result["phase_binding"], _PHASE_BINDING_FIELDS, "phase_binding")
    canonical_phase_schema = canonicalize_static_proxy_phase_schema(phase["phase_schema"])
    if phase["phase_schema"] != canonical_phase_schema:
        raise ValueError("coverage proxy phase schema is not canonical")
    if (
        _require_sha256(phase["phase_schema_fingerprint"], "phase_schema_fingerprint")
        != canonical_phase_schema["phase_schema_fingerprint"]
    ):
        raise ValueError("coverage proxy phase schema fingerprint mismatch")
    required_phases = _int_list(phase["required_phase_ids"], "required_phase_ids")
    observed_phases = _int_list(phase["observed_phase_ids"], "observed_phase_ids")
    if required_phases != sorted(set(required_phases)) or observed_phases != sorted(set(observed_phases)):
        raise ValueError("coverage proxy phase inventories must be sorted and unique")
    if not set(required_phases).issubset(observed_phases):
        raise ValueError("coverage proxy is missing required phases")
    phase_counts = _phase_count_mapping(phase["per_phase_sample_counts"])
    if observed_phases != sorted(phase_counts):
        raise ValueError("coverage proxy observed phases differ from phase counts")

    selection = _strict_mapping(result["selection"], _SELECTION_FIELDS, "selection")
    if selection["kind"] != "all_qc_passed_target_control_frames":
        raise ValueError("coverage proxy selection kind is unsupported")
    min_samples = _strict_int(selection["min_phase_samples"], "min_phase_samples")
    if min_samples <= 0 or any(phase_counts.get(phase_id, 0) < min_samples for phase_id in required_phases):
        raise ValueError("coverage proxy required phase sample threshold is not satisfied")
    thresholds = _strict_mapping(
        selection["thresholds"],
        _SELECTION_THRESHOLD_FIELDS,
        "selection thresholds",
    )
    for field in _SELECTION_THRESHOLD_FIELDS:
        _unit_fraction(thresholds[field], field)

    proxy = _strict_mapping(result["proxy_binding"], _PROXY_BINDING_FIELDS, "proxy_binding")
    if proxy["filename"] != COVERAGE_PROXY_FILENAME:
        raise ValueError("coverage proxy filename is not canonical")
    _require_sha256(proxy["file_sha256"], "proxy file_sha256")
    _require_sha256(proxy["content_fingerprint"], "proxy content_fingerprint")
    shape = proxy["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
        or shape[1] != action_dim
        or sum(phase_counts.values()) != shape[0]
    ):
        raise ValueError("coverage proxy shape is inconsistent")
    if proxy["dtype"] != "float32":
        raise ValueError("coverage proxy dtype must be float32")
    identity_fields = proxy["identity_fields"]
    if (
        not isinstance(identity_fields, list)
        or identity_fields != sorted(set(identity_fields))
        or not set(identity_fields).issubset(_IDENTITY_FIELDS)
    ):
        raise ValueError("coverage proxy identity_fields are invalid")
    _require_sha256(proxy["identity_fingerprint"], "identity_fingerprint")
    supplied = _require_sha256(result["manifest_fingerprint"], "manifest_fingerprint")
    expected = _json_fingerprint(result, excluded=("manifest_fingerprint",))
    if supplied != expected:
        raise ValueError("coverage proxy manifest fingerprint mismatch")
    return result


def _validate_embedded_source_binding(
    manifest: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    qc: Mapping[str, Any],
) -> None:
    """Cross-check embedded canonical evidence against every proxy binding."""

    binding = manifest["source_binding"]
    expected_source_values = {
        "source_manifest_schema_version": source["schema_version"],
        "source_manifest_fingerprint": source["manifest_fingerprint"],
        "source_qc_schema_version": qc["schema_version"],
        "source_qc_fingerprint": qc["qc_fingerprint"],
        "source_kind": source["source_kind"],
        "source_npz_sha256": source["source_npz_sha256"],
        "reference_fingerprint": source["reference_fingerprint"],
        "phase_annotation_fingerprint": source["phase_annotation_fingerprint"],
        "checkpoint_fingerprint": source["checkpoint_fingerprint"],
        "optimizer_fingerprint": source["optimizer_fingerprint"],
    }
    for field, expected in expected_source_values.items():
        if binding[field] != expected:
            raise ValueError(f"coverage proxy source binding {field} differs from embedded evidence")
    if source["target_skill_id"] != manifest["target_skill_id"] or qc["target_skill_id"] != manifest["target_skill_id"]:
        raise ValueError("coverage proxy target skill differs from embedded source/QC")
    physical = manifest["physical_binding"]
    expected_physical_values = {
        "action_dim": source["action_dim"],
        "model_hash": source["model_hash"],
        "actuator_schema_hash": source["actuator_schema_hash"],
        "ctrlrange_schema_hash": source["ctrlrange_schema_hash"],
    }
    for field, expected in expected_physical_values.items():
        if physical[field] != expected:
            raise ValueError(f"coverage proxy physical binding {field} differs from embedded source")
    required_true = (
        "passed",
        "tracking_qc_passed",
        "forward_replay_verified",
        "complete_trajectory_coverage",
    )
    failed = [field for field in required_true if qc[field] is not True]
    if failed:
        raise ValueError(f"coverage proxy embedded source QC is not formal/passable: {failed}")


def _load_control_arrays(path: Path, *, expected_action_dim: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        raw_fields = [field for field in _RAW_CONTROL_FIELDS if field in data.files]
        if len(raw_fields) != 1:
            if not raw_fields and ({"qpos", "qvel"} & set(data.files)):
                raise ValueError("kinematic qpos/qvel cannot be used as physical excitation")
            raise ValueError(
                "target-control NPZ requires exactly one raw applied-control field: "
                "teacher_ctrl_physical or applied_ctrl"
            )
        ctrlrange_fields = [field for field in _CTRLRANGE_FIELDS if field in data.files]
        if len(ctrlrange_fields) != 1:
            raise ValueError("target-control NPZ requires exactly one ordered ctrlrange field")
        contract_fields = {
            "physical_signal_schema_version",
            "muscle_excitation_transform",
            "muscle_channel_contract_schema_version",
            "actuator_ids",
            "actuator_dyntype",
            "actuator_actnum",
            "actuator_actadr",
            "model_na",
        }
        missing = sorted(
            {"actuator_names", "phase_id", *contract_fields} - set(data.files)
        )
        if missing:
            raise ValueError(f"target-control NPZ is missing required fields: {missing}")
        raw = np.asarray(data[raw_fields[0]], dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != int(expected_action_dim) or raw.shape[0] < 2:
            raise ValueError(
                f"raw applied controls must have shape [T,{int(expected_action_dim)}] with T>=2, got {raw.shape}"
            )
        if not np.all(np.isfinite(raw)):
            raise ValueError("raw applied controls contain NaN/Inf")
        names = _actuator_names(data["actuator_names"], expected_width=expected_action_dim)
        ctrlrange = validate_ordered_ctrlrange(names, data[ctrlrange_fields[0]])
        if (
            _npz_scalar_string(
                data["physical_signal_schema_version"],
                "physical_signal_schema_version",
            )
            != PHYSICAL_SIGNAL_SCHEMA_VERSION
            or _npz_scalar_string(
                data["muscle_excitation_transform"],
                "muscle_excitation_transform",
            )
            != UNIT_EXCITATION_TRANSFORM
        ):
            raise ValueError("target-control NPZ uses a legacy physical signal schema/transform")
        channel_contract = validate_muscle_channel_contract(
            {
                "schema_version": _npz_scalar_string(
                    data["muscle_channel_contract_schema_version"],
                    "muscle_channel_contract_schema_version",
                ),
                "actuator_names": list(names),
                "actuator_ids": np.asarray(data["actuator_ids"]).tolist(),
                "actuator_dyntype": [
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in np.asarray(data["actuator_dyntype"]).tolist()
                ],
                "actuator_actnum": np.asarray(data["actuator_actnum"]).tolist(),
                "actuator_actadr": np.asarray(data["actuator_actadr"]).tolist(),
                "model_na": int(np.asarray(data["model_na"]).item()),
            },
            expected_names=names,
        )
        phases = _validate_phase_id(data["phase_id"], sample_count=raw.shape[0])
        declared_excitation = (
            None if "physical_excitation" not in data.files else np.asarray(data["physical_excitation"])
        )
        identity: dict[str, np.ndarray] = {}
        for field in _IDENTITY_FIELDS:
            if field not in data.files:
                continue
            value = np.asarray(data[field])
            if value.shape != (raw.shape[0],) or not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"optional identity field {field!r} must be integer [T]")
            if np.any(value < 0):
                raise ValueError(f"optional identity field {field!r} must be non-negative")
            identity[field] = value.astype(np.int64, copy=False)
    return {
        "raw_field": raw_fields[0],
        "raw_ctrl": raw,
        "actuator_names": names,
        "actuator_ctrlrange": ctrlrange,
        "phase_id": phases,
        "declared_excitation": declared_excitation,
        "muscle_channel_contract": channel_contract.to_metadata(),
        "identity": identity,
    }


def _full_file_content_fingerprint(path: str | Path) -> dict[str, Any]:
    """Audit every byte of one optimizer file or directory."""

    supplied = str(path)
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        root = resolved
        files = sorted(
            (item for item in resolved.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    elif resolved.is_file():
        root = resolved.parent
        files = [resolved]
    else:
        raise FileNotFoundError(f"control artifact does not exist: {resolved}")
    if not files:
        raise ValueError(f"control artifact contains no files: {resolved}")
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for item in files:
        relative = item.relative_to(root).as_posix()
        payload = item.read_bytes()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        total_bytes += len(payload)
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "num_bytes": len(payload),
            }
        )
    return {
        "schema_version": "checkpoint_content_fingerprint_v1",
        "supplied_path": supplied,
        "resolved_path": str(resolved),
        "sha256": digest.hexdigest(),
        "num_files": len(records),
        "num_bytes": int(total_bytes),
        "files": records,
    }


def _validate_content_audit(
    value: Any,
    *,
    source_kind: str,
    verify_live_artifact: bool,
) -> dict[str, Any]:
    audit = _strict_mapping(value, _CONTENT_AUDIT_FIELDS, "target-control artifact audit")
    if audit["schema_version"] != "checkpoint_content_fingerprint_v1":
        raise ValueError("target-control artifact audit schema is unsupported")
    _nonempty_string(audit["supplied_path"], "artifact audit supplied_path")
    resolved_path = _nonempty_string(audit["resolved_path"], "artifact audit resolved_path")
    if not Path(resolved_path).is_absolute():
        raise ValueError("target-control artifact audit resolved_path must be absolute")
    _require_sha256(audit["sha256"], "artifact audit sha256")
    num_files = _strict_int(audit["num_files"], "artifact audit num_files")
    num_bytes = _strict_int(audit["num_bytes"], "artifact audit num_bytes")
    files = audit["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("target-control artifact audit files must be a non-empty list")
    canonical_files: list[dict[str, Any]] = []
    for index, raw in enumerate(files):
        item = _strict_mapping(raw, _CONTENT_AUDIT_FILE_FIELDS, f"artifact audit files[{index}]")
        relative = _nonempty_string(item["path"], f"artifact audit files[{index}].path")
        parsed = PurePosixPath(relative)
        if (
            parsed.is_absolute()
            or relative != parsed.as_posix()
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise ValueError("target-control artifact audit file paths must be canonical and relative")
        item_bytes = _strict_int(item["num_bytes"], f"artifact audit files[{index}].num_bytes")
        if item_bytes < 0:
            raise ValueError("target-control artifact audit file sizes cannot be negative")
        canonical_files.append(
            {
                "path": relative,
                "sha256": _require_sha256(item["sha256"], f"artifact audit files[{index}].sha256"),
                "num_bytes": item_bytes,
            }
        )
    if canonical_files != sorted(canonical_files, key=lambda item: item["path"]):
        raise ValueError("target-control artifact audit files must be sorted by path")
    if len({item["path"] for item in canonical_files}) != len(canonical_files):
        raise ValueError("target-control artifact audit file paths must be unique")
    if num_files != len(canonical_files) or num_bytes != sum(item["num_bytes"] for item in canonical_files):
        raise ValueError("target-control artifact audit inventory totals are inconsistent")
    audit["files"] = canonical_files
    if verify_live_artifact:
        try:
            actual_path = Path(resolved_path)
            actual = (
                checkpoint_content_fingerprint(actual_path, canonicalize=False)
                if source_kind == "full_action_teacher" and actual_path.is_dir()
                else _full_file_content_fingerprint(actual_path)
            )
        except (OSError, ValueError) as exc:
            raise ValueError("target-control live artifact is missing or unreadable") from exc
        actual = _validate_content_audit(
            actual,
            source_kind=source_kind,
            verify_live_artifact=False,
        )
        identity_fields = ("sha256", "num_files", "num_bytes", "files")
        if any(audit[field] != actual[field] for field in identity_fields):
            raise ValueError("target-control live artifact content drifted from source manifest audit")
    return audit


def _target_control_qc_checks(
    *,
    tracking_qc_passed: bool,
    forward_replay_verified: bool,
    complete_trajectory_coverage: bool,
    early_termination_rate: float,
    frame_coverage: float,
    episode_success_rate: float,
    excitation_upper_saturation_fraction: float,
    thresholds: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "tracking_qc_passed": tracking_qc_passed is True,
        "forward_replay_verified": forward_replay_verified is True,
        "complete_trajectory_coverage": complete_trajectory_coverage is True,
        "early_termination_rate": float(early_termination_rate) <= float(thresholds["max_early_termination_rate"]),
        "frame_coverage": float(frame_coverage) >= float(thresholds["min_frame_coverage"]),
        "episode_success_rate": float(episode_success_rate) >= float(thresholds["min_episode_success_rate"]),
        "excitation_upper_saturation_fraction": float(excitation_upper_saturation_fraction)
        <= float(thresholds["max_excitation_upper_saturation_fraction"]),
    }


def _validate_qc_against_arrays(
    qc: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    phase_counts: Mapping[int, int],
    excitation: np.ndarray,
    max_early_termination_rate: float,
    min_frame_coverage: float,
    min_episode_success_rate: float,
    max_excitation_upper_saturation_fraction: float,
) -> None:
    if qc["passed"] is not True:
        raise ValueError("target-control source QC did not pass")
    if qc["tracking_qc_passed"] is not True:
        raise ValueError("target-control tracking QC did not pass")
    if qc["forward_replay_verified"] is not True:
        raise ValueError("target-control source lacks verified physics forward replay")
    if qc["complete_trajectory_coverage"] is not True:
        raise ValueError("target-control source does not cover complete target phases")
    if float(qc["early_termination_rate"]) > _unit_fraction(
        max_early_termination_rate,
        "max_early_termination_rate",
    ):
        raise ValueError("target-control early-termination rate exceeds threshold")
    if float(qc["frame_coverage"]) < _unit_fraction(min_frame_coverage, "min_frame_coverage"):
        raise ValueError("target-control frame coverage is below threshold")
    if float(qc["episode_success_rate"]) < _unit_fraction(
        min_episode_success_rate,
        "min_episode_success_rate",
    ):
        raise ValueError("target-control episode success rate is below threshold")
    max_saturation = _unit_fraction(
        max_excitation_upper_saturation_fraction,
        "max_excitation_upper_saturation_fraction",
    )
    computed_saturation = float(np.mean(excitation >= 1.0 - UNIT_INTERVAL_TOLERANCE))
    declared_saturation = float(qc["excitation_upper_saturation_fraction"])
    if not math.isclose(computed_saturation, declared_saturation, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("target-control QC excitation saturation differs from source controls")
    if computed_saturation > max_saturation:
        raise ValueError("target-control excitation upper saturation exceeds threshold")
    declared_counts = _phase_count_mapping(qc["per_phase_sample_counts"])
    if dict(phase_counts) != declared_counts:
        raise ValueError("target-control QC phase counts differ from source controls")
    if sorted(phase_counts) != qc["observed_phase_ids"]:
        raise ValueError("target-control QC observed phases differ from source controls")
    if source["source_kind"] not in _SOURCE_KINDS:
        raise ValueError("target-control source is not eligible for coverage")


def _validate_phase_semantics(phase_id: np.ndarray, phase_schema: Mapping[str, Any]) -> None:
    canonical = canonicalize_static_proxy_phase_schema(phase_schema)
    declared = {int(item["id"]) for item in canonical["phases"]}
    observed = {int(value) for value in np.unique(phase_id).tolist()}
    unknown = sorted(observed - declared)
    if unknown:
        raise ValueError(f"target-control phase ids are absent from semantic phase schema: {unknown}")


def _canonical_required_phases(
    values: Sequence[int],
    phase_schema: Mapping[str, Any],
) -> tuple[int, ...]:
    required = tuple(sorted(_strict_int(value, "required phase id") for value in values))
    if not required or len(required) != len(set(required)) or any(value < 0 for value in required):
        raise ValueError("required_phase_ids must be unique non-negative integers")
    declared = {int(item["id"]) for item in phase_schema["phases"]}
    unknown = sorted(set(required) - declared)
    if unknown:
        raise ValueError(f"required phases are absent from semantic phase schema: {unknown}")
    return required


def _resolve_proxy_manifest_path(path: str | Path) -> Path:
    source = Path(path)
    if source.is_dir():
        source = source / COVERAGE_PROXY_MANIFEST_FILENAME
    elif source.name == COVERAGE_PROXY_FILENAME:
        source = source.with_name(COVERAGE_PROXY_MANIFEST_FILENAME)
    return _existing_file(source, "coverage proxy manifest").resolve()


def _identity_fingerprint(identity: Mapping[str, np.ndarray], sample_count: int) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema_version": "coverage_proxy_identity_v1",
                "sample_count": int(sample_count),
                "fields": sorted(identity),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for field in sorted(identity):
        values = np.asarray(identity[field])
        if values.shape != (int(sample_count),) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"identity field {field!r} must be integer [T]")
        digest.update(field.encode("utf-8"))
        digest.update(np.ascontiguousarray(values, dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()


def _phase_counts(phase_id: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(np.asarray(phase_id, dtype=np.int64), return_counts=True)
    return {int(value): int(count) for value, count in zip(values.tolist(), counts.tolist(), strict=True)}


def _phase_count_mapping(value: Any) -> dict[int, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("per_phase_sample_counts must be a non-empty object")
    result: dict[int, int] = {}
    for key, raw_count in value.items():
        try:
            phase_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError("per-phase sample-count keys must be canonical integers") from exc
        if str(phase_id) != str(key) or phase_id < 0 or phase_id in result:
            raise ValueError("per-phase sample-count keys must be unique canonical non-negative integers")
        count = _strict_int(raw_count, f"per_phase_sample_counts[{key!r}]")
        if count <= 0:
            raise ValueError("per-phase sample counts must be positive")
        result[phase_id] = count
    return dict(sorted(result.items()))


def _validate_phase_id(values: Any, *, sample_count: int | None = None) -> np.ndarray:
    phases = np.asarray(values)
    if phases.ndim != 1 or (sample_count is not None and phases.shape != (int(sample_count),)):
        raise ValueError("phase_id must be rank-1 and match target-control samples")
    if not np.issubdtype(phases.dtype, np.integer) or np.issubdtype(phases.dtype, np.bool_):
        raise ValueError("phase_id must use an integer dtype; truncation is forbidden")
    if phases.size == 0 or np.any(phases < 0):
        raise ValueError("phase_id must be non-empty and non-negative")
    return phases.astype(np.int32, copy=False)


def _actuator_names(values: Any, *, expected_width: int) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape != (int(expected_width),):
        raise ValueError(f"actuator_names must have shape ({int(expected_width)},)")
    names = tuple(str(value) for value in array.tolist())
    if any(not value for value in names) or len(set(names)) != len(names):
        raise ValueError("actuator_names must be non-empty and unique")
    return names


def _strict_mapping(value: Any, fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    result = dict(value)
    missing = sorted(fields - set(result))
    unknown = sorted(set(result) - fields)
    if missing or unknown:
        raise ValueError(f"{name} fields differ from contract: missing={missing} unknown={unknown}")
    return result


def _int_list(value: Any, field: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list")
    return [_strict_int(item, field) for item in value]


def _strict_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _strict_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return bool(value)


def _unit_fraction(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite fraction")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must lie in [0,1]")
    return result


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _npz_scalar_string(value: Any, field: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field} must be a scalar string")
    item = array.item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _require_sha256(value: Any, field: str) -> str:
    result = str(value or "")
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{field} must be a lowercase 64-hex SHA-256")
    return result


def _optional_sha256(value: Any, field: str) -> str | None:
    return None if value is None else _require_sha256(value, field)


def _existing_file(path: str | Path, name: str) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(f"{name} does not exist: {result}")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_fingerprint(payload: Mapping[str, Any], *, excluded: Sequence[str]) -> str:
    canonical = {str(key): value for key, value in payload.items() if key not in set(excluded)}
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal audited full-action ChinaJump controls into a content-addressed excitation proxy."
    )
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-qc", required=True)
    parser.add_argument("--phase-schema", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-target-skill-id", default="ChinaJump")
    parser.add_argument("--expected-action-dim", type=int, default=354)
    parser.add_argument("--required-phase-id", type=int, action="append", default=[])
    parser.add_argument("--min-phase-samples", type=int, default=2)
    parser.add_argument("--max-early-termination-rate", type=float, default=0.05)
    parser.add_argument("--min-frame-coverage", type=float, default=0.95)
    parser.add_argument("--min-episode-success-rate", type=float, default=0.95)
    parser.add_argument("--max-excitation-upper-saturation-fraction", type=float, default=0.05)
    parser.add_argument("--expected-model-hash", default=None)
    parser.add_argument("--expected-reference-fingerprint", default=None)
    parser.add_argument("--expected-control-source-fingerprint", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    artifact = build_coverage_proxy(
        args.input_npz,
        source_manifest_path=args.source_manifest,
        source_qc_path=args.source_qc,
        phase_schema_path=args.phase_schema,
        output_dir=args.output_dir,
        expected_target_skill_id=args.expected_target_skill_id,
        expected_action_dim=args.expected_action_dim,
        required_phase_ids=tuple(args.required_phase_id or (1, 2, 3, 4)),
        min_phase_samples=args.min_phase_samples,
        max_early_termination_rate=args.max_early_termination_rate,
        min_frame_coverage=args.min_frame_coverage,
        min_episode_success_rate=args.min_episode_success_rate,
        max_excitation_upper_saturation_fraction=(args.max_excitation_upper_saturation_fraction),
        expected_model_hash=args.expected_model_hash,
        expected_reference_fingerprint=args.expected_reference_fingerprint,
        expected_control_source_fingerprint=args.expected_control_source_fingerprint,
    )
    print(
        json.dumps(
            {
                "manifest": str(artifact.manifest_path),
                "manifest_fingerprint": artifact.manifest_fingerprint,
                "proxy": str(artifact.npz_path),
                "content_fingerprint": artifact.content_fingerprint,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
