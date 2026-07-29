"""Versioned action contracts shared by direct and muscle-synergy policies.

The policy action dimension is not enough to identify a controller.  A valid
checkpoint also depends on the ordered body actuators and, for a synergy
policy, every frozen decoder artifact.  The contract therefore exposes two
identities instead of conflating decoder semantics with one stage's model:

* ``portable_decoder_core_fingerprint`` identifies the ordered body-action ABI
  and frozen decoder artifacts that must remain stable across Stage 1, racket
  mass adaptation, distillation, and Stage 3.
* ``stage_runtime_binding_fingerprint`` identifies one concrete runtime model,
  physical action interface, and stage-specific coverage evidence.

The full ``contract_fingerprint`` binds both layers for exact checkpoint
resume.  Cross-stage hand-off deliberately compares only the portable layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.synergy.schema import ctrlrange_schema_hash

BODY_SYNERGY_CONTRACT_SCHEMA_VERSION = "body_synergy_contract_v2"
PORTABLE_DECODER_CORE_SCHEMA_VERSION = "body_synergy_portable_decoder_core_v1"
STAGE_RUNTIME_BINDING_SCHEMA_VERSION = "body_synergy_stage_runtime_binding_v1"
PORTABLE_COMPATIBILITY = "portable_decoder_core"
EXACT_RUNTIME_COMPATIBILITY = "exact_runtime"
SUPPORTED_COMPATIBILITY_LEVELS = (
    PORTABLE_COMPATIBILITY,
    EXACT_RUNTIME_COMPATIBILITY,
)
FULL_354_ACTION_SCHEMA_VERSION = "full_354_action_v2"
EARLY_SYNERGY_ACTION_SCHEMA_VERSION = "early_synergy_action_v2"

FULL_354_MODE = "full_354"
FIXED_SYNERGY_MODE = "fixed_synergy"
FIXED_SYNERGY_RESIDUAL_MODE = "fixed_synergy_residual"
SUPPORTED_ACTION_MODES = (
    FULL_354_MODE,
    FIXED_SYNERGY_MODE,
    FIXED_SYNERGY_RESIDUAL_MODE,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_sha256(payload: Any) -> str:
    """Hash a finite JSON value with one repository-independent encoding."""

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def canonical_action_mode(config_or_mode: Any = None) -> str:
    """Resolve legacy action settings to an explicit canonical mode.

    Missing configuration and the historical mode-less ``enabled: false``
    setting are direct full-action control.  ``enabled: true`` without a mode
    preserves the former early-synergy default.  When both keys are explicit
    they must agree, so a contradictory configuration cannot silently train a
    different action space.
    """

    if isinstance(config_or_mode, str):
        raw_mode = config_or_mode
        enabled = None
    else:
        raw_mode = _cfg_get(config_or_mode, "mode", None)
        enabled = _cfg_get(config_or_mode, "enabled", None)

    if raw_mode in (None, ""):
        return FIXED_SYNERGY_MODE if enabled is True else FULL_354_MODE

    aliases = {
        "direct": FULL_354_MODE,
        "full": FULL_354_MODE,
        "full_action": FULL_354_MODE,
        "end_to_end": FULL_354_MODE,
        "synergy": FIXED_SYNERGY_MODE,
        "synergy_residual": FIXED_SYNERGY_RESIDUAL_MODE,
    }
    mode = aliases.get(str(raw_mode).strip().lower(), str(raw_mode).strip().lower())
    if mode not in SUPPORTED_ACTION_MODES:
        raise ValueError(
            f"unsupported action_representation.mode={raw_mode!r}; expected one of {SUPPORTED_ACTION_MODES}"
        )
    if enabled is not None:
        expected_enabled = mode != FULL_354_MODE
        if enabled is not expected_enabled:
            raise ValueError(
                "action_representation.mode and legacy enabled switch conflict: "
                f"mode={mode!r} requires enabled={expected_enabled}"
            )
    return mode


def build_full_354_action_manifest(
    *,
    actuator_names: Sequence[str],
    ctrlrange: np.ndarray,
    runtime_model_hash: str | None,
    source_binding: Mapping[str, Any] | None = None,
    policy_action_dim: int | None = None,
) -> dict[str, Any]:
    """Build the content-bound manifest for the direct action head.

    No identity basis is created: policy coordinates are the ordered body
    actuator coordinates themselves. The mode name is a dimensional contract,
    not a generic synonym for direct control; legacy 416-D fixtures remain on
    their explicitly separate compatibility path.
    """

    names = _validated_names(actuator_names)
    body_dim = len(names)
    if body_dim != 354:
        raise ValueError(f"full_354 requires exactly 354 ordered actuators, got {body_dim}")
    policy_dim = body_dim if policy_action_dim is None else int(policy_action_dim)
    if policy_dim != body_dim:
        raise ValueError(
            "full_354 policy action dimension must equal the ordered body actuator dimension: "
            f"policy={policy_dim} body={body_dim}"
        )
    ranges = np.asarray(ctrlrange, dtype=np.float64)
    if ranges.shape != (body_dim, 2) or not np.all(np.isfinite(ranges)):
        raise ValueError(f"full_354 ctrlrange must have finite shape ({body_dim}, 2)")
    if not np.array_equal(ranges, np.tile([0.0, 1.0], (body_dim, 1))):
        raise ValueError(
            "full_354_action_v2 requires every verified MuJoCo muscle "
            "actuator ctrlrange to be exactly [0,1]"
        )
    control_hash = ctrlrange_schema_hash(names, ranges)
    model_hash = _optional_sha256(runtime_model_hash, "runtime_model_hash")
    source = dict(source_binding or {"kind": "runtime_direct_body_action_interface"})

    unsigned = {
        "schema_version": FULL_354_ACTION_SCHEMA_VERSION,
        "mode": FULL_354_MODE,
        "policy_action_dim": policy_dim,
        "body_action_dim": body_dim,
        "actuator_names": list(names),
        "actuator_schema_hash": actuator_schema_hash(names),
        "control_range_hash": control_hash,
        "runtime_control_range_hash": control_hash,
        "runtime_model_hash": model_hash,
        "runtime_binding_complete": model_hash is not None,
        "effective_excitation_semantics": "clip(raw_data_ctrl,0,1)",
        "basis_rank": 0,
        "basis_fingerprint": None,
        "runtime_basis_fingerprint": None,
        "coefficient_transform_fingerprint": None,
        "coefficient_statistics_fingerprint": None,
        "tonic_baseline_fingerprint": None,
        "residual_dim": 0,
        "residual_basis_fingerprint": None,
        "residual_fit_contract_fingerprint": None,
        "residual_allowed_muscle_mask_fingerprint": None,
        "residual_alpha": 0.0,
        "source_binding": _json_copy(source),
        "coverage_gate": None,
        "target_coverage_evidence": {
            "status": "not_applicable_full_354",
            "reason": "direct policy spans the complete ordered body action ABI",
        },
    }
    return {
        **unsigned,
        "physical_action_interface_hash": canonical_json_sha256(unsigned),
    }


@dataclass(frozen=True)
class BodySynergyContractV2:
    """Two-level identity of one body-action representation.

    Decoder/core fields are portable between stages.  Runtime model, physical
    interface, and coverage fields describe only the stage that materialized
    this instance.  The split is computed from validated semantic fields; no
    caller-supplied sub-fingerprint is trusted.
    """

    mode: str
    body_action_dim: int
    policy_action_dim: int
    actuator_names: tuple[str, ...]
    actuator_schema_hash: str
    control_range_hash: str
    runtime_control_range_hash: str
    runtime_model_hash: str | None
    physical_action_interface_hash: str
    basis_fingerprint: str | None = None
    runtime_basis_fingerprint: str | None = None
    basis_rank: int = 0
    coefficient_transform_fingerprint: str | None = None
    coefficient_statistics_fingerprint: str | None = None
    tonic_baseline_fingerprint: str | None = None
    residual_basis_fingerprint: str | None = None
    residual_fit_contract_fingerprint: str | None = None
    residual_allowed_muscle_mask_fingerprint: str | None = None
    residual_dim: int = 0
    residual_alpha: float = 0.0
    source_binding_json: str = "{}"
    coverage_binding_json: str = "{}"

    def __post_init__(self) -> None:
        mode = canonical_action_mode(self.mode)
        names = _validated_names(self.actuator_names)
        body_dim = int(self.body_action_dim)
        policy_dim = int(self.policy_action_dim)
        basis_rank = int(self.basis_rank)
        residual_dim = int(self.residual_dim)
        residual_alpha = float(self.residual_alpha)
        if body_dim <= 0 or policy_dim <= 0 or len(names) != body_dim:
            raise ValueError("body/policy dimensions must be positive and match ordered actuator names")
        if actuator_schema_hash(names) != _require_sha256(
            self.actuator_schema_hash, "actuator_schema_hash"
        ):
            raise ValueError("BodySynergyContractV2 ordered actuator schema hash mismatch")

        control_hash = _require_sha256(self.control_range_hash, "control_range_hash")
        runtime_control_hash = _require_sha256(
            self.runtime_control_range_hash, "runtime_control_range_hash"
        )
        if runtime_control_hash != control_hash:
            raise ValueError("formal and runtime control-range hashes differ")
        model_hash = _optional_sha256(self.runtime_model_hash, "runtime_model_hash")
        interface_hash = _require_sha256(
            self.physical_action_interface_hash, "physical_action_interface_hash"
        )

        basis_fingerprint = _optional_sha256(self.basis_fingerprint, "basis_fingerprint")
        runtime_basis_fingerprint = _optional_sha256(
            self.runtime_basis_fingerprint, "runtime_basis_fingerprint"
        )
        coefficient_transform = _optional_sha256(
            self.coefficient_transform_fingerprint,
            "coefficient_transform_fingerprint",
        )
        coefficient_statistics = _optional_sha256(
            self.coefficient_statistics_fingerprint,
            "coefficient_statistics_fingerprint",
        )
        tonic = _optional_sha256(self.tonic_baseline_fingerprint, "tonic_baseline_fingerprint")
        residual_basis = _optional_sha256(
            self.residual_basis_fingerprint, "residual_basis_fingerprint"
        )
        residual_fit = _optional_sha256(
            self.residual_fit_contract_fingerprint,
            "residual_fit_contract_fingerprint",
        )
        residual_mask = _optional_sha256(
            self.residual_allowed_muscle_mask_fingerprint,
            "residual_allowed_muscle_mask_fingerprint",
        )
        if not math.isfinite(residual_alpha) or residual_alpha < 0.0:
            raise ValueError("residual_alpha must be finite and non-negative")

        if mode == FULL_354_MODE:
            if body_dim != 354:
                raise ValueError("full_354 contract requires body_action_dim == 354")
            if policy_dim != body_dim:
                raise ValueError("full_354 contract requires policy_action_dim == body_action_dim")
            if any(
                value is not None
                for value in (
                    basis_fingerprint,
                    runtime_basis_fingerprint,
                    coefficient_transform,
                    coefficient_statistics,
                    tonic,
                    residual_basis,
                    residual_fit,
                    residual_mask,
                )
            ) or basis_rank != 0 or residual_dim != 0 or residual_alpha != 0.0:
                raise ValueError("full_354 contract must not carry synergy or residual artifacts")
        else:
            if basis_rank <= 0 or basis_rank >= body_dim:
                raise ValueError("fixed-synergy basis rank must lie strictly between zero and body dimension")
            if basis_fingerprint is None or runtime_basis_fingerprint is None:
                raise ValueError("fixed-synergy contract requires formal and runtime W fingerprints")
            if coefficient_transform is None or coefficient_statistics is None or tonic is None:
                raise ValueError("fixed-synergy contract requires coefficient-transform/statistics and tonic fingerprints")
            if mode == FIXED_SYNERGY_MODE:
                if policy_dim != basis_rank:
                    raise ValueError("fixed_synergy policy dimension must equal W rank")
                if any(value is not None for value in (residual_basis, residual_fit, residual_mask)):
                    raise ValueError("fixed_synergy contract must not carry residual artifacts")
                if residual_dim != 0 or residual_alpha != 0.0:
                    raise ValueError("fixed_synergy contract requires a zero-dimensional residual")
            else:
                if residual_dim <= 0 or policy_dim != basis_rank + residual_dim:
                    raise ValueError("fixed_synergy_residual policy dimension must equal W rank plus residual rank")
                if residual_basis is None or residual_mask is None or residual_alpha <= 0.0:
                    raise ValueError("fixed_synergy_residual contract requires R, its allowed mask, and positive alpha")

        source_json = _canonical_json_text(self.source_binding_json, "source_binding_json")
        coverage_json = _canonical_json_text(self.coverage_binding_json, "coverage_binding_json")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "body_action_dim", body_dim)
        object.__setattr__(self, "policy_action_dim", policy_dim)
        object.__setattr__(self, "actuator_names", names)
        object.__setattr__(self, "actuator_schema_hash", actuator_schema_hash(names))
        object.__setattr__(self, "control_range_hash", control_hash)
        object.__setattr__(self, "runtime_control_range_hash", runtime_control_hash)
        object.__setattr__(self, "runtime_model_hash", model_hash)
        object.__setattr__(self, "physical_action_interface_hash", interface_hash)
        object.__setattr__(self, "basis_fingerprint", basis_fingerprint)
        object.__setattr__(self, "runtime_basis_fingerprint", runtime_basis_fingerprint)
        object.__setattr__(self, "basis_rank", basis_rank)
        object.__setattr__(self, "coefficient_transform_fingerprint", coefficient_transform)
        object.__setattr__(self, "coefficient_statistics_fingerprint", coefficient_statistics)
        object.__setattr__(self, "tonic_baseline_fingerprint", tonic)
        object.__setattr__(self, "residual_basis_fingerprint", residual_basis)
        object.__setattr__(self, "residual_fit_contract_fingerprint", residual_fit)
        object.__setattr__(self, "residual_allowed_muscle_mask_fingerprint", residual_mask)
        object.__setattr__(self, "residual_dim", residual_dim)
        object.__setattr__(self, "residual_alpha", residual_alpha)
        object.__setattr__(self, "source_binding_json", source_json)
        object.__setattr__(self, "coverage_binding_json", coverage_json)

    @property
    def source_binding(self) -> Any:
        return json.loads(self.source_binding_json)

    @property
    def coverage_binding(self) -> Any:
        return json.loads(self.coverage_binding_json)

    @property
    def portable_decoder_core_fingerprint(self) -> str:
        """Identity preserved when the same decoder moves to another stage."""

        return canonical_json_sha256(self._portable_decoder_core_payload())

    @property
    def stage_runtime_binding_fingerprint(self) -> str:
        """Identity of this stage's model/interface/coverage binding."""

        return canonical_json_sha256(self._stage_runtime_binding_payload())

    @property
    def runtime_binding_complete(self) -> bool:
        """Whether exact runtime validation can bind a concrete MuJoCo model."""

        return self.runtime_model_hash is not None

    @property
    def contract_fingerprint(self) -> str:
        return canonical_json_sha256(self._unsigned_manifest())

    @classmethod
    def from_action_manifest(
        cls,
        action_manifest: Mapping[str, Any],
        *,
        actuator_names: Sequence[str] | None = None,
    ) -> BodySynergyContractV2:
        """Derive and validate a v2 contract from a direct or early-synergy manifest."""

        manifest = _json_copy(dict(action_manifest))
        supplied_interface_hash = _require_sha256(
            manifest.get("physical_action_interface_hash"),
            "physical_action_interface_hash",
        )
        unsigned_interface = {
            key: value
            for key, value in manifest.items()
            if key not in {"physical_action_interface_hash", "exploration"}
        }
        if canonical_json_sha256(unsigned_interface) != supplied_interface_hash:
            raise ValueError("action manifest physical_action_interface_hash mismatch")

        mode = canonical_action_mode(manifest.get("mode"))
        if (
            mode == FULL_354_MODE
            and manifest.get("schema_version") != FULL_354_ACTION_SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported full_354 action schema_version; legacy signed-control "
                "manifests must be migrated and retrained"
            )
        if (
            mode in {FIXED_SYNERGY_MODE, FIXED_SYNERGY_RESIDUAL_MODE}
            and manifest.get("schema_version")
            != EARLY_SYNERGY_ACTION_SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported early-synergy action schema_version; v1 decoder "
                "artifacts must be regenerated from a v2 excitation basis"
            )
        names_value = manifest.get("actuator_names") if actuator_names is None else actuator_names
        if names_value is None:
            raise ValueError("action manifest is missing ordered actuator_names")
        names = _validated_names(names_value)
        body_dim = int(manifest.get("body_action_dim", -1))
        if len(names) != body_dim:
            raise ValueError("action manifest actuator_names length differs from body_action_dim")
        supplied_schema_hash = _require_sha256(
            manifest.get("actuator_schema_hash"), "actuator_schema_hash"
        )
        if supplied_schema_hash != actuator_schema_hash(names):
            raise ValueError("action manifest ordered actuator schema hash mismatch")

        source = manifest.get("source_binding")
        if source is None:
            source = {
                "basis_source": manifest.get("basis_source"),
                "primitive_source_binding": manifest.get("primitive_source_binding"),
            }
            if "frozen_body_decoder_execution_binding" in manifest:
                source["frozen_body_decoder_execution_binding"] = manifest[
                    "frozen_body_decoder_execution_binding"
                ]
        coverage = {
            "coverage_gate": manifest.get("coverage_gate"),
            "target_coverage_evidence": manifest.get("target_coverage_evidence"),
        }
        return cls(
            mode=mode,
            body_action_dim=body_dim,
            policy_action_dim=int(manifest.get("policy_action_dim", -1)),
            actuator_names=names,
            actuator_schema_hash=supplied_schema_hash,
            control_range_hash=manifest.get("control_range_hash"),
            runtime_control_range_hash=manifest.get(
                "runtime_control_range_hash", manifest.get("control_range_hash")
            ),
            runtime_model_hash=manifest.get("runtime_model_hash"),
            physical_action_interface_hash=supplied_interface_hash,
            basis_fingerprint=manifest.get("basis_fingerprint"),
            runtime_basis_fingerprint=manifest.get("runtime_basis_fingerprint"),
            basis_rank=int(manifest.get("basis_rank", 0)),
            coefficient_transform_fingerprint=manifest.get("coefficient_transform_fingerprint"),
            coefficient_statistics_fingerprint=manifest.get("coefficient_statistics_fingerprint"),
            tonic_baseline_fingerprint=manifest.get("tonic_baseline_fingerprint"),
            residual_basis_fingerprint=manifest.get("residual_basis_fingerprint"),
            residual_fit_contract_fingerprint=manifest.get("residual_fit_contract_fingerprint"),
            residual_allowed_muscle_mask_fingerprint=manifest.get(
                "residual_allowed_muscle_mask_fingerprint"
            ),
            residual_dim=int(manifest.get("residual_dim", 0)),
            residual_alpha=float(manifest.get("residual_alpha", 0.0)),
            source_binding_json=_canonical_json_value(source),
            coverage_binding_json=_canonical_json_value(coverage),
        )

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> BodySynergyContractV2:
        """Load a serialized contract and verify both identity layers.

        The two sub-fingerprints are mandatory even though they are derivable.
        This makes a partially upgraded or hand-edited artifact fail closed
        instead of silently falling back to the historical monolithic hash.
        """

        payload = _json_copy(dict(manifest))
        expected_fields = set(cls._manifest_field_names()) | {
            "schema_version",
            "portable_decoder_core_fingerprint",
            "stage_runtime_binding_fingerprint",
            "contract_fingerprint",
        }
        if set(payload) != expected_fields:
            missing = sorted(expected_fields - set(payload))
            extra = sorted(set(payload) - expected_fields)
            raise ValueError(f"BodySynergyContractV2 fields differ from schema: missing={missing} extra={extra}")
        if payload.get("schema_version") != BODY_SYNERGY_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported body synergy contract schema_version")
        supplied_fingerprint = _require_sha256(
            payload.pop("contract_fingerprint"), "contract_fingerprint"
        )
        supplied_portable_fingerprint = _require_sha256(
            payload.pop("portable_decoder_core_fingerprint"),
            "portable_decoder_core_fingerprint",
        )
        supplied_runtime_fingerprint = _require_sha256(
            payload.pop("stage_runtime_binding_fingerprint"),
            "stage_runtime_binding_fingerprint",
        )
        payload.pop("schema_version")
        source_binding = payload.pop("source_binding")
        coverage_binding = payload.pop("coverage_binding")
        contract = cls(
            **{
                **payload,
                "actuator_names": tuple(payload["actuator_names"]),
                "source_binding_json": _canonical_json_value(source_binding),
                "coverage_binding_json": _canonical_json_value(coverage_binding),
            }
        )
        if contract.portable_decoder_core_fingerprint != supplied_portable_fingerprint:
            raise ValueError(
                "BodySynergyContractV2 portable_decoder_core_fingerprint mismatch"
            )
        if contract.stage_runtime_binding_fingerprint != supplied_runtime_fingerprint:
            raise ValueError(
                "BodySynergyContractV2 stage_runtime_binding_fingerprint mismatch"
            )
        if contract.contract_fingerprint != supplied_fingerprint:
            raise ValueError("BodySynergyContractV2 contract_fingerprint mismatch")
        return contract

    def to_manifest(self) -> dict[str, Any]:
        unsigned = self._unsigned_manifest()
        return {**unsigned, "contract_fingerprint": canonical_json_sha256(unsigned)}

    def save(self, path: str | Path) -> Path:
        return save_body_synergy_contract(path, self)

    def assert_portable_compatible(self, other: BodySynergyContractV2) -> None:
        """Require the same ordered ABI and frozen decoder semantics.

        Runtime model hashes and coverage reports are intentionally ignored.
        This is the compatibility level for Stage-1 -> Stage-2 -> distillation
        -> Stage-3 hand-off of an unchanged ``W``/transform/tonic/``R`` core.
        """

        self._require_contract(other)
        if (
            self.portable_decoder_core_fingerprint
            == other.portable_decoder_core_fingerprint
        ):
            return
        differences = _payload_differences(
            self._portable_decoder_core_payload(),
            other._portable_decoder_core_payload(),
        )
        raise ValueError(
            "incompatible BodySynergyContractV2 portable decoder cores; "
            f"differing fields={differences}"
        )

    def assert_exact_runtime_compatible(
        self,
        other: BodySynergyContractV2,
        *,
        require_complete: bool = False,
    ) -> None:
        """Require both the portable core and the exact stage runtime binding."""

        self._require_contract(other)
        self.assert_portable_compatible(other)
        if require_complete and not (
            self.runtime_binding_complete and other.runtime_binding_complete
        ):
            raise ValueError(
                "exact runtime compatibility requires concrete runtime model hashes"
            )
        if (
            self.stage_runtime_binding_fingerprint
            == other.stage_runtime_binding_fingerprint
        ):
            return
        differences = _payload_differences(
            self._stage_runtime_binding_payload(),
            other._stage_runtime_binding_payload(),
        )
        raise ValueError(
            "incompatible BodySynergyContractV2 exact runtime bindings; "
            f"differing fields={differences}"
        )

    def assert_compatible(self, other: BodySynergyContractV2) -> None:
        """Backward-compatible name for exact stage-runtime compatibility."""

        self.assert_exact_runtime_compatible(other)

    @staticmethod
    def _require_contract(other: BodySynergyContractV2) -> None:
        if not isinstance(other, BodySynergyContractV2):
            raise TypeError("other must be a BodySynergyContractV2")

    def validate_runtime(
        self,
        *,
        actuator_names: Sequence[str],
        ctrlrange: np.ndarray,
        runtime_model_hash: str | None,
        mode: str | None = None,
        policy_action_dim: int | None = None,
        physical_action_interface_hash: str | None = None,
        require_model_hash: bool = False,
    ) -> None:
        """Fail closed when a runtime no longer matches its persisted contract."""

        names = _validated_names(actuator_names)
        if names != self.actuator_names:
            raise ValueError("runtime ordered actuator names differ from BodySynergyContractV2")
        ranges = np.asarray(ctrlrange, dtype=np.float64)
        if ranges.shape != (self.body_action_dim, 2) or not np.all(np.isfinite(ranges)):
            raise ValueError("runtime ctrlrange shape/content is invalid")
        runtime_control_hash = ctrlrange_schema_hash(names, ranges)
        if runtime_control_hash != self.runtime_control_range_hash:
            raise ValueError("runtime control-range hash differs from BodySynergyContractV2")
        if mode is not None and canonical_action_mode(mode) != self.mode:
            raise ValueError("runtime action mode differs from BodySynergyContractV2")
        if policy_action_dim is not None and int(policy_action_dim) != self.policy_action_dim:
            raise ValueError("runtime policy action dimension differs from BodySynergyContractV2")
        if physical_action_interface_hash is not None and (
            _require_sha256(physical_action_interface_hash, "physical_action_interface_hash")
            != self.physical_action_interface_hash
        ):
            raise ValueError("runtime physical action interface hash differs from BodySynergyContractV2")

        actual_model_hash = _optional_sha256(runtime_model_hash, "runtime_model_hash")
        if self.runtime_model_hash is None:
            if require_model_hash or actual_model_hash is not None:
                raise ValueError("contract has no model binding but runtime model validation was requested")
        elif actual_model_hash != self.runtime_model_hash:
            raise ValueError("runtime MuJoCo model hash differs from BodySynergyContractV2")

    def _portable_decoder_core_payload(self) -> dict[str, Any]:
        """Canonical payload whose meaning must survive stage/model changes."""

        return {
            "schema_version": PORTABLE_DECODER_CORE_SCHEMA_VERSION,
            "mode": self.mode,
            "body_action_dim": self.body_action_dim,
            "policy_action_dim": self.policy_action_dim,
            "actuator_names": list(self.actuator_names),
            "actuator_schema_hash": self.actuator_schema_hash,
            # The formal decoder transform is portable.  The separately stored
            # runtime control-range hash belongs to the stage binding below.
            "control_range_hash": self.control_range_hash,
            "basis_fingerprint": self.basis_fingerprint,
            "runtime_basis_fingerprint": self.runtime_basis_fingerprint,
            "basis_rank": self.basis_rank,
            "coefficient_transform_fingerprint": self.coefficient_transform_fingerprint,
            "coefficient_statistics_fingerprint": self.coefficient_statistics_fingerprint,
            "tonic_baseline_fingerprint": self.tonic_baseline_fingerprint,
            "residual_basis_fingerprint": self.residual_basis_fingerprint,
            "residual_fit_contract_fingerprint": self.residual_fit_contract_fingerprint,
            "residual_allowed_muscle_mask_fingerprint": (
                self.residual_allowed_muscle_mask_fingerprint
            ),
            "residual_dim": self.residual_dim,
            "residual_alpha": self.residual_alpha,
            "source_binding": self.source_binding,
        }

    def _stage_runtime_binding_payload(self) -> dict[str, Any]:
        """Canonical payload for one stage's concrete execution context."""

        return {
            "schema_version": STAGE_RUNTIME_BINDING_SCHEMA_VERSION,
            # Prevent a valid runtime binding from being transplanted onto a
            # different W/decoder core.
            "portable_decoder_core_fingerprint": (
                self.portable_decoder_core_fingerprint
            ),
            "runtime_control_range_hash": self.runtime_control_range_hash,
            "runtime_model_hash": self.runtime_model_hash,
            "physical_action_interface_hash": self.physical_action_interface_hash,
            "coverage_binding": self.coverage_binding,
        }

    @classmethod
    def _manifest_field_names(cls) -> tuple[str, ...]:
        return (
            "mode",
            "body_action_dim",
            "policy_action_dim",
            "actuator_names",
            "actuator_schema_hash",
            "control_range_hash",
            "runtime_control_range_hash",
            "runtime_model_hash",
            "physical_action_interface_hash",
            "basis_fingerprint",
            "runtime_basis_fingerprint",
            "basis_rank",
            "coefficient_transform_fingerprint",
            "coefficient_statistics_fingerprint",
            "tonic_baseline_fingerprint",
            "residual_basis_fingerprint",
            "residual_fit_contract_fingerprint",
            "residual_allowed_muscle_mask_fingerprint",
            "residual_dim",
            "residual_alpha",
            "source_binding",
            "coverage_binding",
        )

    def _unsigned_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": BODY_SYNERGY_CONTRACT_SCHEMA_VERSION,
            "mode": self.mode,
            "body_action_dim": self.body_action_dim,
            "policy_action_dim": self.policy_action_dim,
            "actuator_names": list(self.actuator_names),
            "actuator_schema_hash": self.actuator_schema_hash,
            "control_range_hash": self.control_range_hash,
            "runtime_control_range_hash": self.runtime_control_range_hash,
            "runtime_model_hash": self.runtime_model_hash,
            "physical_action_interface_hash": self.physical_action_interface_hash,
            "basis_fingerprint": self.basis_fingerprint,
            "runtime_basis_fingerprint": self.runtime_basis_fingerprint,
            "basis_rank": self.basis_rank,
            "coefficient_transform_fingerprint": self.coefficient_transform_fingerprint,
            "coefficient_statistics_fingerprint": self.coefficient_statistics_fingerprint,
            "tonic_baseline_fingerprint": self.tonic_baseline_fingerprint,
            "residual_basis_fingerprint": self.residual_basis_fingerprint,
            "residual_fit_contract_fingerprint": self.residual_fit_contract_fingerprint,
            "residual_allowed_muscle_mask_fingerprint": self.residual_allowed_muscle_mask_fingerprint,
            "residual_dim": self.residual_dim,
            "residual_alpha": self.residual_alpha,
            "source_binding": self.source_binding,
            "coverage_binding": self.coverage_binding,
            "portable_decoder_core_fingerprint": (
                self.portable_decoder_core_fingerprint
            ),
            "stage_runtime_binding_fingerprint": (
                self.stage_runtime_binding_fingerprint
            ),
        }


def save_body_synergy_contract(
    path: str | Path,
    contract: BodySynergyContractV2,
) -> Path:
    """Write one canonical JSON contract."""

    if not isinstance(contract, BodySynergyContractV2):
        raise TypeError("contract must be a BodySynergyContractV2")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(contract.to_manifest(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def load_body_synergy_contract(path: str | Path) -> BodySynergyContractV2:
    """Load a canonical JSON contract while rejecting duplicate object keys."""

    source = Path(path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in BodySynergyContractV2: {key}")
            result[key] = value
        return result

    payload = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(payload, Mapping):
        raise ValueError("BodySynergyContractV2 file must contain one JSON object")
    return BodySynergyContractV2.from_manifest(payload)


def load_compatible_body_synergy_contract(
    path: str | Path,
    expected: BodySynergyContractV2 | Mapping[str, Any],
    *,
    compatibility: str = EXACT_RUNTIME_COMPATIBILITY,
    require_complete_runtime: bool = False,
) -> BodySynergyContractV2:
    """Load a contract and validate it at one explicit compatibility level.

    This is a generic artifact helper only; it does not pretend that Stage-2 or
    latent checkpoints already consume the contract.  Callers must explicitly
    choose portable hand-off or exact-runtime resume semantics.
    """

    loaded = load_body_synergy_contract(path)
    reference = (
        expected
        if isinstance(expected, BodySynergyContractV2)
        else BodySynergyContractV2.from_manifest(expected)
    )
    level = str(compatibility)
    if level == PORTABLE_COMPATIBILITY:
        reference.assert_portable_compatible(loaded)
    elif level == EXACT_RUNTIME_COMPATIBILITY:
        reference.assert_exact_runtime_compatible(
            loaded,
            require_complete=require_complete_runtime,
        )
    else:
        raise ValueError(
            f"unsupported BodySynergyContractV2 compatibility={level!r}; "
            f"expected one of {SUPPORTED_COMPATIBILITY_LEVELS}"
        )
    return loaded


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _payload_differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    keys = set(left) | set(right)
    return sorted(key for key in keys if left.get(key) != right.get(key))


def _validated_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if not names or any(not name for name in names):
        raise ValueError("ordered actuator_names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("ordered actuator_names contain duplicates")
    return names


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _optional_sha256(value: Any, label: str) -> str | None:
    if value in (None, ""):
        return None
    return _require_sha256(value, label)


def _canonical_json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_json_text(value: str, label: str) -> str:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must contain canonical JSON") from exc
    return _canonical_json_value(parsed)


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


__all__ = [
    "BODY_SYNERGY_CONTRACT_SCHEMA_VERSION",
    "EARLY_SYNERGY_ACTION_SCHEMA_VERSION",
    "EXACT_RUNTIME_COMPATIBILITY",
    "FIXED_SYNERGY_MODE",
    "FIXED_SYNERGY_RESIDUAL_MODE",
    "FULL_354_ACTION_SCHEMA_VERSION",
    "FULL_354_MODE",
    "PORTABLE_COMPATIBILITY",
    "PORTABLE_DECODER_CORE_SCHEMA_VERSION",
    "STAGE_RUNTIME_BINDING_SCHEMA_VERSION",
    "SUPPORTED_ACTION_MODES",
    "SUPPORTED_COMPATIBILITY_LEVELS",
    "BodySynergyContractV2",
    "build_full_354_action_manifest",
    "canonical_action_mode",
    "canonical_json_sha256",
    "load_body_synergy_contract",
    "load_compatible_body_synergy_contract",
    "save_body_synergy_contract",
]
