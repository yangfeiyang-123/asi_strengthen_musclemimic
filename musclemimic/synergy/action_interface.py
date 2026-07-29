"""Frozen early-synergy action contracts for Stage-1 policies.

The Stage-1 policy emits unconstrained, low-dimensional Gaussian actions.  A
fixed, content-bound decoder maps those actions to unit muscle excitation and
then to the existing normalized body-action ABI.  Nothing in this module is
trainable; changing any decoder input changes the persisted action manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.distill.physical import (
    MUSCLE_EXCITATION_FORMULA,
    MUSCLE_EXCITATION_ROUNDOFF_POLICY,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
    validate_muscle_channel_contract,
    validate_unit_muscle_ctrlrange,
)
from musclemimic.latent_muscle.synergy_decoder import (
    LoadedSynergyBasis,
    load_fixed_synergy_basis,
    validate_decoder_synergy_basis,
)
from musclemimic.synergy.frozen_decoder import (
    FrozenBodyDecoder,
    bounded_synergy_coefficients,
    build_frozen_body_decoder_execution_binding,
)
from musclemimic.synergy.hybrid_basis import (
    HybridBasisResult,
    validate_hybrid_basis_result,
)
from musclemimic.synergy.multistage_contract import BodySynergyContractV2
from musclemimic.synergy.rank_selection import (
    candidate_basis_fingerprint,
    validate_dynamic_coverage_gate,
    validate_dynamic_coverage_requirement,
)
from musclemimic.synergy.schema import EXCITATION_SIGNAL_KIND, ctrlrange_schema_hash

ACTION_SCHEMA_VERSION = "early_synergy_action_v2"
COEFFICIENT_STATS_SCHEMA_VERSION = "early_synergy_coefficient_stats_v2"
RESIDUAL_BASIS_SCHEMA_VERSION = "early_synergy_residual_basis_v3"
_STRICT_SELECTION_REASON = "smallest_rank_meeting_all_vaf_and_stability_gates"


class EarlySynergyActionOutput(NamedTuple):
    """Decoded action and the quantities needed for diagnostics/rollout export."""

    body_action: Any
    physical_excitation: Any
    preclip_excitation: Any
    synergy_excitation: Any
    synergy_coefficients: Any
    residual_coefficients: Any
    residual_excitation: Any


@dataclass(frozen=True)
class CoefficientTransform:
    """Bounded, non-negative coefficient mapping fixed before PPO starts."""

    kind: str
    maximum: np.ndarray
    center: np.ndarray
    temperature: np.ndarray
    fingerprint: str
    source_fingerprint: str

    @property
    def dimension(self) -> int:
        return int(self.maximum.shape[0])

    @property
    def bias(self) -> np.ndarray:
        ratio = np.clip(self.center / self.maximum, 1e-6, 1.0 - 1e-6)
        return np.log(ratio) - np.log1p(-ratio)

    def apply(self, raw_action: Any) -> Any:
        raw = jnp.asarray(raw_action)
        return bounded_synergy_coefficients(
            raw,
            maximum=self.maximum,
            bias=self.bias,
            temperature=self.temperature,
        )

    def derivative_at_zero(self) -> np.ndarray:
        ratio = np.clip(self.center / self.maximum, 1e-6, 1.0 - 1e-6)
        return self.maximum * ratio * (1.0 - ratio) / self.temperature


@dataclass(frozen=True)
class StructuredResidualBasis:
    """Small signed residual basis; never a full-dimensional raw bypass."""

    basis: np.ndarray
    actuator_names: tuple[str, ...]
    fingerprint: str
    source_basis_fingerprint: str
    source_path: str
    normalization: str
    row_l1_operator_max: float
    allowed_muscle_mask: np.ndarray
    fit_contract: dict[str, Any] | None

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[1])


@dataclass(frozen=True)
class EarlySynergyActionInterface:
    """Runtime-ready, immutable Stage-1 action representation."""

    mode: str
    basis: LoadedSynergyBasis
    formal_basis_fingerprint: str
    coefficient_transform: CoefficientTransform
    tonic_baseline: np.ndarray
    tonic_baseline_fingerprint: str
    residual_basis: StructuredResidualBasis | None
    residual_alpha: float
    action_manifest: dict[str, Any]
    body_synergy_contract: BodySynergyContractV2
    frozen_decoder: FrozenBodyDecoder

    @property
    def body_action_dim(self) -> int:
        return self.basis.action_dim

    @property
    def synergy_dim(self) -> int:
        return self.basis.synergy_dim

    @property
    def residual_dim(self) -> int:
        return 0 if self.residual_basis is None else self.residual_basis.dimension

    @property
    def policy_action_dim(self) -> int:
        return self.synergy_dim + self.residual_dim

    @property
    def decoder_jacobian_at_zero(self) -> np.ndarray:
        coefficient_jacobian = (
            np.asarray(self.basis.basis, dtype=np.float64) * (self.coefficient_transform.derivative_at_zero()[None, :])
        )
        if self.residual_basis is None:
            return coefficient_jacobian
        residual_jacobian = float(self.residual_alpha) * np.asarray(self.residual_basis.basis, dtype=np.float64)
        return np.concatenate([coefficient_jacobian, residual_jacobian], axis=1)

    def decode(self, policy_action: Any) -> EarlySynergyActionOutput:
        output = self.frozen_decoder.decode(policy_action)
        return EarlySynergyActionOutput(
            body_action=output.body_action,
            physical_excitation=output.physical_excitation,
            preclip_excitation=output.preclip_excitation,
            synergy_excitation=output.synergy_excitation,
            synergy_coefficients=output.synergy_coefficients,
            residual_coefficients=output.residual_coefficients,
            residual_excitation=output.residual_excitation,
        )

    def metrics(self, output: EarlySynergyActionOutput) -> dict[str, Any]:
        """Return JIT-safe scalar diagnostics for one decoded action batch."""

        coefficients = jnp.asarray(output.synergy_coefficients)
        maximum = jnp.asarray(self.coefficient_transform.maximum, dtype=coefficients.dtype)
        normalized_coefficients = coefficients / maximum
        coefficient_energy = jnp.sum(jnp.square(coefficients), axis=-1)
        coefficient_effective_dim = jnp.square(jnp.sum(jnp.abs(coefficients), axis=-1)) / (coefficient_energy + 1e-8)

        excitation = jnp.asarray(output.physical_excitation)
        preclip_excitation = jnp.asarray(output.preclip_excitation)
        bounds = jnp.asarray(self.basis.excitation_bounds, dtype=excitation.dtype)
        width = bounds[:, 1] - bounds[:, 0]
        normalized_excitation = (excitation - bounds[:, 0]) / width
        residual = jnp.asarray(output.residual_excitation)
        excitation_energy = jnp.sum(jnp.square(excitation), axis=-1)
        residual_energy = jnp.sum(jnp.square(residual), axis=-1)
        clip_correction = excitation - preclip_excitation

        def mean(value):
            return jnp.mean(jnp.asarray(value, dtype=jnp.float32))

        return {
            "synergy_coefficient_mean": mean(coefficients),
            "synergy_coefficient_max": jnp.max(coefficients),
            "synergy_coefficient_saturation_fraction": mean(
                jnp.logical_or(normalized_coefficients <= 0.01, normalized_coefficients >= 0.99)
            ),
            "synergy_coefficient_effective_dimension": mean(coefficient_effective_dim),
            "synergy_decoded_excitation_mean": mean(excitation),
            "synergy_decoded_excitation_rms": jnp.sqrt(mean(jnp.square(excitation))),
            "synergy_decoded_excitation_saturation_fraction": mean(
                jnp.logical_or(normalized_excitation <= 0.01, normalized_excitation >= 0.99)
            ),
            "synergy_preclip_out_of_bounds_fraction": mean(
                jnp.logical_or(
                    preclip_excitation < bounds[:, 0],
                    preclip_excitation > bounds[:, 1],
                )
            ),
            "synergy_preclip_excitation_rms": jnp.sqrt(
                mean(jnp.square(preclip_excitation))
            ),
            "synergy_clip_correction_rms": jnp.sqrt(
                mean(jnp.square(clip_correction))
            ),
            "synergy_residual_l1": mean(jnp.sum(jnp.abs(residual), axis=-1)),
            "synergy_residual_l2": mean(jnp.sqrt(residual_energy + 1e-12)),
            "synergy_residual_energy_fraction": mean(residual_energy / (excitation_energy + 1e-8)),
        }


def jax_sigmoid(value: Any) -> Any:
    """Numerically stable sigmoid kept as a tiny public test seam."""

    return jnp.asarray(0.5, dtype=jnp.asarray(value).dtype) * (jnp.tanh(jnp.asarray(value) * 0.5) + 1.0)


def save_coefficient_statistics(
    path: str | Path,
    coefficients: np.ndarray,
    *,
    basis_fingerprint: str,
) -> dict[str, Any]:
    """Persist train-only coefficient quantiles bound to one formal basis."""

    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("coefficient statistics require a non-empty [samples,rank] matrix")
    if not np.all(np.isfinite(values)) or np.any(values < -1e-10):
        raise ValueError("coefficient statistics require finite non-negative coefficients")
    values = np.maximum(values, 0.0)
    payload = {
        "schema_version": COEFFICIENT_STATS_SCHEMA_VERSION,
        "basis_fingerprint": _require_sha256(basis_fingerprint, "basis_fingerprint"),
        "sample_count": int(values.shape[0]),
        "rank": int(values.shape[1]),
        "coefficient_q01": np.quantile(values, 0.01, axis=0),
        "coefficient_q05": np.quantile(values, 0.05, axis=0),
        "coefficient_q50": np.quantile(values, 0.50, axis=0),
        "coefficient_q95": np.quantile(values, 0.95, axis=0),
        "coefficient_q99": np.quantile(values, 0.99, axis=0),
        "coefficient_max": np.max(values, axis=0),
        "coefficient_mean": np.mean(values, axis=0),
        "coefficient_std": np.std(values, axis=0),
    }
    fingerprint = _coefficient_stats_fingerprint(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        **{key: np.asarray(value) for key, value in payload.items()},
        stats_fingerprint=np.asarray(fingerprint),
    )
    return {**payload, "stats_fingerprint": fingerprint, "path": str(destination.resolve())}


def load_coefficient_statistics(
    path: str | Path,
    *,
    expected_basis_fingerprint: str,
    expected_rank: int,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"coefficient statistics do not exist: {source}")
    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    schema = _scalar_string(payload.pop("schema_version", None))
    if schema != COEFFICIENT_STATS_SCHEMA_VERSION:
        raise ValueError("unsupported early-synergy coefficient statistics schema")
    basis_fingerprint = _scalar_string(payload.pop("basis_fingerprint", None))
    if basis_fingerprint != str(expected_basis_fingerprint):
        raise ValueError("coefficient statistics are bound to a different synergy basis")
    supplied_fingerprint = _scalar_string(payload.pop("stats_fingerprint", None))
    sample_count = _scalar_int(payload.pop("sample_count", None), "sample_count")
    rank = _scalar_int(payload.pop("rank", None), "rank")
    if sample_count <= 0 or rank != int(expected_rank):
        raise ValueError("coefficient statistics sample count/rank is incompatible")
    native: dict[str, Any] = {
        "schema_version": schema,
        "basis_fingerprint": basis_fingerprint,
        "sample_count": sample_count,
        "rank": rank,
    }
    required = {
        "coefficient_q01",
        "coefficient_q05",
        "coefficient_q50",
        "coefficient_q95",
        "coefficient_q99",
        "coefficient_max",
        "coefficient_mean",
        "coefficient_std",
    }
    if set(payload) != required:
        raise ValueError(f"coefficient statistics fields differ from contract: {sorted(payload)}")
    for key, raw in payload.items():
        value = np.asarray(raw, dtype=np.float64)
        if value.shape != (rank,) or not np.all(np.isfinite(value)) or np.any(value < -1e-10):
            raise ValueError(f"coefficient statistics {key} must be finite non-negative shape ({rank},)")
        native[key] = np.maximum(value, 0.0)
    actual_fingerprint = _coefficient_stats_fingerprint(native)
    if supplied_fingerprint != actual_fingerprint:
        raise ValueError("coefficient statistics fingerprint mismatch")
    return {
        **native,
        "stats_fingerprint": actual_fingerprint,
        "path": str(source.resolve()),
    }


def save_structured_residual_basis(
    path: str | Path,
    *,
    basis: np.ndarray,
    actuator_names: Sequence[str],
    source_basis_fingerprint: str,
    source_description: str,
    allowed_muscle_mask: Sequence[bool] | np.ndarray | None = None,
    fit_contract: Mapping[str, Any] | None = None,
) -> StructuredResidualBasis:
    """Persist a small signed residual basis with exact row-order provenance."""

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    source_matrix, names = _validate_residual_matrix(basis, actuator_names)
    active_rows = np.any(np.abs(source_matrix) > 1e-12, axis=1)
    if allowed_muscle_mask is None:
        allowed_mask = active_rows
    else:
        supplied_mask = np.asarray(allowed_muscle_mask)
        if supplied_mask.dtype != np.bool_ or supplied_mask.shape != (len(names),):
            raise ValueError("structured residual allowed_muscle_mask must be boolean [body_action_dim]")
        allowed_mask = supplied_mask.astype(bool, copy=True)
    if not np.any(allowed_mask) or np.any(active_rows & ~allowed_mask):
        raise ValueError("structured residual has nonzero rows outside its non-empty allowed-muscle mask")
    source_column_norms = np.linalg.norm(source_matrix, axis=0)
    # Alpha only has stable physical meaning when every residual coordinate
    # uses the same scale.  Store unit-L2 columns rather than accepting an
    # arbitrary amplitude hidden inside R.
    matrix = (source_matrix / source_column_norms[None, :]).astype(np.float32)
    canonical_fit_contract = (
        None
        if fit_contract is None
        else _validate_residual_fit_contract(
            fit_contract,
            source_basis_fingerprint=source_basis_fingerprint,
            actuator_names=names,
            allowed_muscle_mask=allowed_mask,
            basis=matrix,
        )
    )
    stored_column_norms = np.linalg.norm(matrix.astype(np.float64), axis=0)
    row_l1_operator_max = float(np.max(np.sum(np.abs(matrix), axis=1)))
    source_fingerprint = _require_sha256(source_basis_fingerprint, "source_basis_fingerprint")
    basis_path = root / "basis.npy"
    np.save(basis_path, matrix.astype(np.float32), allow_pickle=False)
    payload = {
        "schema_version": RESIDUAL_BASIS_SCHEMA_VERSION,
        "basis_file": basis_path.name,
        "basis_sha256": _file_sha256(basis_path),
        "basis_shape": list(matrix.shape),
        "actuator_names": list(names),
        "actuator_schema_hash": actuator_schema_hash(names),
        "source_basis_fingerprint": source_fingerprint,
        "source_description": str(source_description).strip(),
        "normalization": "unit_l2_columns",
        "source_column_l2_norms": source_column_norms.tolist(),
        "stored_column_l2_norms": stored_column_norms.tolist(),
        "row_l1_operator_max": row_l1_operator_max,
        "allowed_muscle_mask": allowed_mask.tolist(),
        "allowed_muscle_mask_fingerprint": _json_sha256(
            {
                "schema_version": "early_synergy_residual_allowed_mask_v1",
                "actuator_names": list(names),
                "allowed_muscle_mask": allowed_mask.tolist(),
            }
        ),
        "fit_contract": canonical_fit_contract,
    }
    if not payload["source_description"]:
        raise ValueError("structured residual basis requires a source description")
    payload["artifact_fingerprint"] = _json_sha256(payload)
    (root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return load_structured_residual_basis(
        root,
        expected_actuator_names=names,
        expected_source_basis_fingerprint=source_fingerprint,
    )


def load_structured_residual_basis(
    path: str | Path,
    *,
    expected_actuator_names: Sequence[str],
    expected_source_basis_fingerprint: str,
) -> StructuredResidualBasis:
    supplied = Path(path)
    manifest_path = supplied if supplied.name == "manifest.json" else supplied / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"structured residual manifest does not exist: {manifest_path}")
    payload = _load_json_object(manifest_path)
    if payload.get("schema_version") != RESIDUAL_BASIS_SCHEMA_VERSION:
        raise ValueError("unsupported structured residual basis schema")
    fingerprint = str(payload.get("artifact_fingerprint", ""))
    unsigned = {key: value for key, value in payload.items() if key != "artifact_fingerprint"}
    if fingerprint != _json_sha256(unsigned):
        raise ValueError("structured residual basis fingerprint mismatch")
    names = tuple(str(name) for name in payload.get("actuator_names", ()))
    expected_names = tuple(str(name) for name in expected_actuator_names)
    if names != expected_names or payload.get("actuator_schema_hash") != actuator_schema_hash(names):
        raise ValueError("structured residual actuator names/order differ from body action schema")
    source_fingerprint = str(payload.get("source_basis_fingerprint", ""))
    if source_fingerprint != str(expected_source_basis_fingerprint):
        raise ValueError("structured residual basis belongs to a different primary basis")
    basis_path = manifest_path.parent / str(payload.get("basis_file", ""))
    if _file_sha256(basis_path) != payload.get("basis_sha256"):
        raise ValueError("structured residual basis file hash mismatch")
    matrix, _ = _validate_residual_matrix(np.load(basis_path, allow_pickle=False), names)
    if list(matrix.shape) != list(payload.get("basis_shape", ())):
        raise ValueError("structured residual basis shape differs from manifest")
    if matrix.shape[1] >= matrix.shape[0]:
        raise ValueError("structured residual basis must remain lower-dimensional than body action")
    if payload.get("normalization") != "unit_l2_columns":
        raise ValueError("structured residual basis must use unit-L2 column normalization")
    stored_norms = np.linalg.norm(matrix, axis=0)
    declared_norms = np.asarray(payload.get("stored_column_l2_norms"), dtype=np.float64)
    if (
        declared_norms.shape != (matrix.shape[1],)
        or not np.allclose(stored_norms, 1.0, rtol=1e-6, atol=1e-7)
        or not np.allclose(declared_norms, stored_norms, rtol=1e-9, atol=1e-9)
    ):
        raise ValueError("structured residual basis column normalization is invalid")
    source_norms = np.asarray(payload.get("source_column_l2_norms"), dtype=np.float64)
    if source_norms.shape != (matrix.shape[1],) or not np.all(np.isfinite(source_norms)) or np.any(source_norms <= 0.0):
        raise ValueError("structured residual source column norms are invalid")
    row_l1_operator_max = float(np.max(np.sum(np.abs(matrix), axis=1)))
    declared_row_l1 = float(payload.get("row_l1_operator_max", float("nan")))
    if not np.isfinite(declared_row_l1) or not np.isclose(
        declared_row_l1,
        row_l1_operator_max,
        rtol=1e-9,
        atol=1e-9,
    ):
        raise ValueError("structured residual row-L1 operator bound is invalid")
    allowed_raw = np.asarray(payload.get("allowed_muscle_mask"))
    if allowed_raw.dtype != np.bool_ or allowed_raw.shape != (len(names),):
        raise ValueError("structured residual allowed-muscle mask is invalid")
    allowed_mask = allowed_raw.astype(bool, copy=True)
    if not np.any(allowed_mask) or np.any(np.any(np.abs(matrix) > 1e-12, axis=1) & ~allowed_mask):
        raise ValueError("structured residual basis violates its allowed-muscle mask")
    mask_fingerprint = _json_sha256(
        {
            "schema_version": "early_synergy_residual_allowed_mask_v1",
            "actuator_names": list(names),
            "allowed_muscle_mask": allowed_mask.tolist(),
        }
    )
    if payload.get("allowed_muscle_mask_fingerprint") != mask_fingerprint:
        raise ValueError("structured residual allowed-muscle mask fingerprint mismatch")
    fit_contract_raw = payload.get("fit_contract")
    fit_contract = (
        None
        if fit_contract_raw is None
        else _validate_residual_fit_contract(
            fit_contract_raw,
            source_basis_fingerprint=source_fingerprint,
            actuator_names=names,
            allowed_muscle_mask=allowed_mask,
            basis=matrix,
        )
    )
    return StructuredResidualBasis(
        basis=matrix.astype(np.float32),
        actuator_names=names,
        fingerprint=fingerprint,
        source_basis_fingerprint=source_fingerprint,
        source_path=str(manifest_path.parent.resolve()),
        normalization="unit_l2_columns",
        row_l1_operator_max=row_l1_operator_max,
        allowed_muscle_mask=allowed_mask,
        fit_contract=fit_contract,
    )


def build_early_synergy_action_interface(
    config: Any,
    *,
    expected_actuator_names: Sequence[str],
    runtime_ctrlrange: np.ndarray | None = None,
    runtime_model_hash: str | None = None,
) -> EarlySynergyActionInterface:
    """Load and validate every frozen artifact used by an early-synergy policy."""

    schema_version = str(_cfg_get(config, "schema_version", ACTION_SCHEMA_VERSION))
    if schema_version != ACTION_SCHEMA_VERSION:
        raise ValueError(f"unsupported early-synergy action schema {schema_version!r}")
    mode = str(_cfg_get(config, "mode", "fixed_synergy"))
    if mode not in {"fixed_synergy", "fixed_synergy_residual"}:
        raise ValueError("early-synergy mode must be fixed_synergy or fixed_synergy_residual")
    if bool(_cfg_get(config, "learned_full_dimensional_baseline", False)):
        raise ValueError("early-synergy Stage-1 forbids a learned full-dimensional baseline")

    names = tuple(str(name) for name in expected_actuator_names)
    expected_dim = int(_cfg_get(config, "expected_underlying_action_dim", len(names)))
    if len(names) != expected_dim:
        raise ValueError(
            f"early-synergy underlying actuator count differs from config: expected={expected_dim} actual={len(names)}"
        )
    actual_schema_hash = actuator_schema_hash(names)
    expected_schema_hash = _cfg_get(config, "expected_actuator_schema_hash", None)
    if expected_schema_hash not in (None, "") and str(expected_schema_hash) != actual_schema_hash:
        raise ValueError("early-synergy actuator schema hash mismatch")

    basis_path = _cfg_get(config, "basis_path", None)
    if not basis_path:
        raise ValueError("early-synergy action representation requires basis_path")
    test_only_legacy = bool(_cfg_get(config, "test_only_allow_legacy_basis", False))
    basis = load_fixed_synergy_basis(
        basis_path,
        expected_actuator_names=names,
        test_only_allow_legacy=test_only_legacy,
    )
    if basis.action_dim != expected_dim:
        raise ValueError("early-synergy basis row count differs from underlying action dimension")
    if basis.synergy_dim >= expected_dim:
        raise ValueError("early-synergy basis rank must be strictly lower than the underlying body-action dimension")
    validate_decoder_synergy_basis(basis, allow_noncanonical=test_only_legacy)
    formal_basis_fingerprint = str(basis.manifest.get("source_fingerprint", basis.fingerprint))
    expected_basis_fingerprint = _cfg_get(config, "expected_basis_fingerprint", None)
    if not test_only_legacy and not expected_basis_fingerprint:
        raise ValueError("production early-synergy config requires expected_basis_fingerprint")
    if expected_basis_fingerprint and str(expected_basis_fingerprint) != formal_basis_fingerprint:
        raise ValueError("early-synergy basis expected fingerprint mismatch")
    runtime_control_range_hash = _validate_excitation_transform(
        basis,
        names,
        runtime_ctrlrange=runtime_ctrlrange,
        require_runtime_binding=bool(_cfg_get(config, "require_runtime_ctrlrange_binding", False)),
    )
    _validate_basis_selection(basis, config)
    primitive_source_binding = _load_primitive_source_contract(
        config,
        basis=basis,
        actuator_schema_hash_value=actual_schema_hash,
        control_range_hash=runtime_control_range_hash,
        runtime_model_hash=runtime_model_hash,
    )

    coefficient_cfg = _cfg_get(config, "coefficient_transform", {}) or {}
    coefficient_transform = _build_coefficient_transform(
        coefficient_cfg,
        basis=basis,
        formal_basis_fingerprint=formal_basis_fingerprint,
        test_only_legacy=test_only_legacy,
    )
    baseline, baseline_fingerprint = _load_tonic_baseline(
        _cfg_get(config, "tonic_baseline", {}) or {},
        action_dim=expected_dim,
        actuator_names=names,
        excitation_bounds=basis.excitation_bounds,
        basis_fingerprint=formal_basis_fingerprint,
    )

    residual_cfg = _cfg_get(config, "residual", {}) or {}
    residual_enabled = bool(_cfg_get(residual_cfg, "enabled", False))
    if (mode == "fixed_synergy_residual") != residual_enabled:
        raise ValueError("early-synergy mode and residual.enabled must agree")
    residual_basis = None
    residual_alpha = 0.0
    residual_schedule_binding = None
    residual_fit_threshold_binding = None
    if residual_enabled:
        residual_path = _cfg_get(residual_cfg, "basis_path", None)
        if not residual_path:
            raise ValueError("fixed_synergy_residual requires residual.basis_path")
        residual_basis = load_structured_residual_basis(
            residual_path,
            expected_actuator_names=names,
            expected_source_basis_fingerprint=formal_basis_fingerprint,
        )
        expected_residual_fingerprint = _cfg_get(residual_cfg, "expected_fingerprint", None)
        if not expected_residual_fingerprint:
            raise ValueError("production structured residual requires expected_fingerprint")
        if str(expected_residual_fingerprint) != residual_basis.fingerprint:
            raise ValueError("structured residual expected fingerprint mismatch")
        residual_alpha = float(_cfg_get(residual_cfg, "alpha", 0.0))
        if not 0.0 < residual_alpha <= 1.0:
            raise ValueError("structured residual alpha must lie in (0,1]")
        schedule_cfg = _cfg_get(residual_cfg, "alpha_schedule", {}) or {}
        if bool(_cfg_get(schedule_cfg, "enabled", False)):
            raise ValueError(
                "Phase-A early-synergy residual does not implement a global-update alpha "
                "schedule; alpha_schedule.enabled must remain false"
            )
        schedule_kind = str(_cfg_get(schedule_cfg, "kind", "constant_phase_a"))
        if schedule_kind != "constant_phase_a":
            raise ValueError("Phase-A structured residual requires constant_phase_a schedule kind")
        schedule_payload = {
            "kind": schedule_kind,
            "enabled": False,
            "alpha": residual_alpha,
        }
        residual_schedule_binding = {
            **schedule_payload,
            "fingerprint": _json_sha256(schedule_payload),
        }
        min_residual_dim = int(_cfg_get(residual_cfg, "min_dimension", 1))
        max_residual_dim = int(_cfg_get(residual_cfg, "max_dimension", 12))
        if not 1 <= min_residual_dim <= max_residual_dim <= 32:
            raise ValueError("structured residual dimension bounds must satisfy 1 <= min <= max <= 32")
        if not min_residual_dim <= residual_basis.dimension <= max_residual_dim:
            raise ValueError(
                "structured residual dimension lies outside the configured low-rank bounds: "
                f"dimension={residual_basis.dimension}, "
                f"allowed=[{min_residual_dim},{max_residual_dim}]"
            )
        max_row_l1_norm = float(_cfg_get(residual_cfg, "max_row_l1_norm", 2.0))
        if not np.isfinite(max_row_l1_norm) or max_row_l1_norm <= 0.0:
            raise ValueError("structured residual max_row_l1_norm must be finite and positive")
        if residual_basis.row_l1_operator_max > max_row_l1_norm:
            raise ValueError(
                "structured residual row-L1 operator bound exceeds the configured maximum: "
                f"actual={residual_basis.row_l1_operator_max:.6g}, "
                f"maximum={max_row_l1_norm:.6g}"
            )
        require_fit_contract = bool(
            primitive_source_binding is not None or _cfg_get(residual_cfg, "require_fit_contract", False)
        )
        fit_contract = residual_basis.fit_contract
        if require_fit_contract and fit_contract is None:
            raise ValueError("primitive structured residual requires a passed train-only fit contract")
        if fit_contract is not None:
            if np.any(np.abs(np.asarray(baseline, dtype=np.float64)) > 1e-12):
                raise ValueError("Phase-A structured residual fit is valid only for the zero tonic baseline")
            if not np.isclose(
                float(fit_contract["reference_alpha"]),
                residual_alpha,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("structured residual fit reference_alpha differs from runtime alpha")
            fit_upper = np.asarray(
                fit_contract["coefficient_upper_bounds"],
                dtype=np.float64,
            )
            if fit_upper.shape != coefficient_transform.maximum.shape or not np.allclose(
                fit_upper,
                coefficient_transform.maximum,
                rtol=1e-6,
                atol=1e-8,
            ):
                raise ValueError("structured residual fit coefficient bounds differ from runtime transform")
            if fit_contract["coefficient_statistics_fingerprint"] != coefficient_transform.source_fingerprint:
                raise ValueError("structured residual fit statistics differ from runtime transform")
            if primitive_source_binding is not None:
                if (
                    fit_contract["source_dataset_fingerprint"] != primitive_source_binding["source_dataset_fingerprint"]
                    or fit_contract["primitive_source_manifest_fingerprint"]
                    != primitive_source_binding["manifest_fingerprint"]
                ):
                    raise ValueError("structured residual fit provenance differs from primitive source")
            required_fit_thresholds = _cfg_get(
                residual_cfg,
                "required_fit_thresholds",
                None,
            )
            if primitive_source_binding is not None and required_fit_thresholds is None:
                raise ValueError("primitive structured residual requires fixed runtime fit thresholds")
            if required_fit_thresholds is not None:
                required_fit_thresholds = dict(required_fit_thresholds)
                threshold_fields = {
                    "min_validation_residual_energy_reduction",
                    "min_group_validation_residual_energy_reduction",
                    "max_validation_coordinate_saturation_fraction",
                }
                if set(required_fit_thresholds) != threshold_fields:
                    raise ValueError("structured residual required_fit_thresholds fields are invalid")
                canonical_thresholds = {
                    field: _bounded_finite_float(
                        required_fit_thresholds[field],
                        f"structured residual required {field}",
                    )
                    for field in sorted(threshold_fields)
                }
                if primitive_source_binding is not None and (
                    canonical_thresholds["min_validation_residual_energy_reduction"] <= 0.0
                    or canonical_thresholds["min_group_validation_residual_energy_reduction"] <= 0.0
                    or canonical_thresholds["max_validation_coordinate_saturation_fraction"] >= 1.0
                ):
                    raise ValueError("primitive residual fit thresholds may not be weakened to trivial gates")
                actual_thresholds = {str(field): float(value) for field, value in fit_contract["thresholds"].items()}
                if any(
                    not np.isclose(
                        actual_thresholds[field],
                        canonical_thresholds[field],
                        rtol=0.0,
                        atol=1e-12,
                    )
                    for field in threshold_fields
                ):
                    raise ValueError("structured residual fit thresholds differ from runtime config")
                residual_fit_threshold_binding = canonical_thresholds

    policy_action_dim = int(basis.synergy_dim + (0 if residual_basis is None else residual_basis.dimension))
    max_policy_action_dim = _cfg_get(config, "max_policy_action_dim", None)
    if max_policy_action_dim is not None:
        max_policy_action_dim = int(max_policy_action_dim)
        if max_policy_action_dim <= 0 or policy_action_dim > max_policy_action_dim:
            raise ValueError(
                "early-synergy policy action dimension exceeds max_policy_action_dim: "
                f"dimension={policy_action_dim}, maximum={max_policy_action_dim}"
            )

    bootstrap_without_target_coverage = bool(_cfg_get(config, "bootstrap_without_target_coverage", False))
    if bootstrap_without_target_coverage and (
        bool(_cfg_get(config, "require_coverage_gate", False)) or bool(_cfg_get(config, "coverage_gate_path", None))
    ):
        raise ValueError(
            "primitive bootstrap must not bind a target coverage gate; use the "
            "formal S0/S1 config once independent target-control evidence exists"
        )
    coverage_gate_binding = _load_coverage_gate_binding(
        config,
        formal_basis_fingerprint=formal_basis_fingerprint,
        coefficient_upper_bounds=coefficient_transform.maximum,
    )
    condition_number, effective_rank = _basis_geometry(basis.basis)
    max_condition = float(_cfg_get(config, "max_basis_condition_number", 100.0))
    min_rank_fraction = float(_cfg_get(config, "min_effective_rank_fraction", 0.8))
    if condition_number > max_condition:
        raise ValueError(f"early-synergy basis condition number {condition_number:.6g} exceeds {max_condition:.6g}")
    rank_fraction = effective_rank / max(1, basis.synergy_dim)
    if rank_fraction < min_rank_fraction:
        raise ValueError(
            f"early-synergy basis effective-rank fraction {rank_fraction:.6g} is below {min_rank_fraction:.6g}"
        )

    control_range_hash = str(basis.manifest["transform"]["ctrlrange_schema_hash"])
    excitation_transform = dict(basis.manifest["transform"])
    channel_contract = validate_muscle_channel_contract(
        excitation_transform.get("muscle_channel_contract"),
        expected_names=names,
    )
    basis_source = {
        "signal_kind": basis.manifest.get("signal_kind"),
        "region": basis.manifest.get("region"),
        "source_dataset_fingerprint": basis.manifest.get("source_dataset_fingerprint"),
        "teacher_checkpoint_fingerprint": basis.manifest.get("teacher_checkpoint_fingerprint"),
        "split_provenance_fingerprint": _json_sha256(basis.manifest.get("split_provenance")),
    }
    frozen_basis = np.asarray(basis.basis, dtype=np.float32)
    frozen_bounds = np.asarray(basis.excitation_bounds, dtype=np.float32)
    frozen_maximum = np.asarray(coefficient_transform.maximum, dtype=np.float32)
    frozen_center = np.asarray(coefficient_transform.center, dtype=np.float32)
    frozen_temperature = np.asarray(
        coefficient_transform.temperature, dtype=np.float32
    )
    frozen_tonic = np.asarray(baseline, dtype=np.float32)
    frozen_residual = (
        np.zeros((expected_dim, 0), dtype=np.float32)
        if residual_basis is None
        else np.asarray(residual_basis.basis, dtype=np.float32)
    )
    residual_basis_fingerprint = (
        None if residual_basis is None else residual_basis.fingerprint
    )
    residual_fit_contract_fingerprint = (
        None
        if residual_basis is None or residual_basis.fit_contract is None
        else residual_basis.fit_contract["fit_contract_fingerprint"]
    )
    residual_allowed_muscle_mask_fingerprint = (
        None
        if residual_basis is None
        else _json_sha256(
            {
                "schema_version": "early_synergy_residual_allowed_mask_v1",
                "actuator_names": list(names),
                "allowed_muscle_mask": residual_basis.allowed_muscle_mask.tolist(),
            }
        )
    )
    decoder_execution_binding = build_frozen_body_decoder_execution_binding(
        mode=mode,
        actuator_names=names,
        residual_alpha=residual_alpha,
        basis=frozen_basis,
        excitation_bounds=frozen_bounds,
        coefficient_maximum=frozen_maximum,
        coefficient_center=frozen_center,
        coefficient_temperature=frozen_temperature,
        tonic_baseline=frozen_tonic,
        residual_basis=frozen_residual,
        basis_fingerprint=formal_basis_fingerprint,
        runtime_basis_fingerprint=basis.fingerprint,
        coefficient_transform_fingerprint=coefficient_transform.fingerprint,
        coefficient_statistics_fingerprint=(
            coefficient_transform.source_fingerprint
        ),
        tonic_baseline_fingerprint=baseline_fingerprint,
        residual_basis_fingerprint=residual_basis_fingerprint,
        residual_fit_contract_fingerprint=residual_fit_contract_fingerprint,
        residual_allowed_muscle_mask_fingerprint=(
            residual_allowed_muscle_mask_fingerprint
        ),
    )
    manifest_without_hash = {
        "schema_version": ACTION_SCHEMA_VERSION,
        "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
        "mode": mode,
        "policy_action_dim": policy_action_dim,
        "body_action_dim": expected_dim,
        "actuator_names": list(names),
        "basis_rank": basis.synergy_dim,
        "residual_dim": 0 if residual_basis is None else residual_basis.dimension,
        "max_policy_action_dim": max_policy_action_dim,
        "basis_fingerprint": formal_basis_fingerprint,
        "runtime_basis_fingerprint": basis.fingerprint,
        "basis_condition_number": condition_number,
        "basis_effective_rank": effective_rank,
        "coefficient_transform_fingerprint": coefficient_transform.fingerprint,
        "coefficient_statistics_fingerprint": coefficient_transform.source_fingerprint,
        "tonic_baseline_fingerprint": baseline_fingerprint,
        "residual_basis_fingerprint": residual_basis_fingerprint,
        "residual_fit_contract_fingerprint": residual_fit_contract_fingerprint,
        "residual_allowed_muscle_mask_fingerprint": (
            residual_allowed_muscle_mask_fingerprint
        ),
        "residual_alpha": residual_alpha,
        "residual_alpha_schedule": residual_schedule_binding,
        "residual_fit_thresholds": residual_fit_threshold_binding,
        "residual_row_l1_operator_max": (None if residual_basis is None else residual_basis.row_l1_operator_max),
        "residual_max_per_muscle_correction": (
            0.0 if residual_basis is None else residual_alpha * residual_basis.row_l1_operator_max
        ),
        "actuator_schema_hash": actual_schema_hash,
        "control_range_hash": control_range_hash,
        "runtime_control_range_hash": runtime_control_range_hash,
        "runtime_model_hash": runtime_model_hash,
        "excitation_transform": excitation_transform,
        "muscle_channel_contract_fingerprint": _json_sha256(
            channel_contract.to_metadata()
        ),
        "frozen_body_decoder_execution_binding": decoder_execution_binding,
        "basis_source": basis_source,
        "primitive_source_binding": primitive_source_binding,
        "coverage_gate": coverage_gate_binding,
        "target_coverage_evidence": {
            "status": (
                "passed_static_proxy"
                if coverage_gate_binding is not None
                else (
                    "not_evaluated_primitive_bootstrap"
                    if bootstrap_without_target_coverage
                    else "not_required_by_config"
                )
            ),
            "bootstrap_without_target_coverage": bootstrap_without_target_coverage,
        },
    }
    action_interface_hash = _json_sha256(manifest_without_hash)
    action_manifest = {
        **manifest_without_hash,
        "physical_action_interface_hash": action_interface_hash,
    }
    expected_interface_hash = _cfg_get(config, "expected_physical_action_interface_hash", None)
    if expected_interface_hash not in (None, "") and str(expected_interface_hash) != action_interface_hash:
        raise ValueError("early-synergy physical action interface hash mismatch")
    body_synergy_contract = BodySynergyContractV2.from_action_manifest(action_manifest)
    frozen_decoder = FrozenBodyDecoder(
        body_synergy_contract=body_synergy_contract,
        basis=frozen_basis,
        excitation_bounds=frozen_bounds,
        coefficient_maximum=frozen_maximum,
        coefficient_center=frozen_center,
        coefficient_temperature=frozen_temperature,
        tonic_baseline=frozen_tonic,
        residual_basis=frozen_residual,
    )
    return EarlySynergyActionInterface(
        mode=mode,
        basis=basis,
        formal_basis_fingerprint=formal_basis_fingerprint,
        coefficient_transform=coefficient_transform,
        tonic_baseline=baseline,
        tonic_baseline_fingerprint=baseline_fingerprint,
        residual_basis=residual_basis,
        residual_alpha=residual_alpha,
        action_manifest=action_manifest,
        body_synergy_contract=body_synergy_contract,
        frozen_decoder=frozen_decoder,
    )


def _build_coefficient_transform(
    config: Any,
    *,
    basis: LoadedSynergyBasis,
    formal_basis_fingerprint: str,
    test_only_legacy: bool,
) -> CoefficientTransform:
    kind = str(_cfg_get(config, "kind", "bounded_sigmoid"))
    if kind != "bounded_sigmoid":
        raise ValueError("early-synergy currently supports only bounded_sigmoid coefficients")
    stats_path = _cfg_get(config, "stats_path", None)
    explicit_maximum = _cfg_get(config, "test_only_maximum", None)
    explicit_center = _cfg_get(config, "test_only_center", None)
    if explicit_maximum is not None or explicit_center is not None:
        if not test_only_legacy or explicit_maximum is None or explicit_center is None:
            raise ValueError("explicit coefficient arrays are test-only and must be supplied together")
        maximum = np.asarray(explicit_maximum, dtype=np.float64)
        center = np.asarray(explicit_center, dtype=np.float64)
        source_fingerprint = _json_sha256({"test_only_maximum": maximum.tolist(), "test_only_center": center.tolist()})
    else:
        if stats_path is None:
            source_path = Path(str(basis.source_path or ""))
            if source_path.name == "manifest.json":
                source_path = source_path.parent
            stats_path = source_path / "coefficient_stats.npz"
        stats = load_coefficient_statistics(
            stats_path,
            expected_basis_fingerprint=formal_basis_fingerprint,
            expected_rank=basis.synergy_dim,
        )
        expected_stats_fingerprint = _cfg_get(
            config,
            "expected_stats_fingerprint",
            None,
        )
        if not test_only_legacy and not expected_stats_fingerprint:
            raise ValueError("production early-synergy coefficient transform requires expected_stats_fingerprint")
        if expected_stats_fingerprint and str(expected_stats_fingerprint) != stats["stats_fingerprint"]:
            raise ValueError("coefficient statistics expected fingerprint mismatch")
        max_source = str(_cfg_get(config, "max_source", "train_q99_times_1p2"))
        if max_source == "train_q99_times_1p2":
            maximum = 1.2 * np.asarray(stats["coefficient_q99"], dtype=np.float64)
        elif max_source == "train_q99":
            maximum = np.asarray(stats["coefficient_q99"], dtype=np.float64)
        else:
            raise ValueError("unsupported coefficient max_source")
        center_source = str(_cfg_get(config, "center_source", "train_q50"))
        if center_source == "train_q50":
            center = np.asarray(stats["coefficient_q50"], dtype=np.float64)
        elif center_source == "train_q05":
            center = np.asarray(stats["coefficient_q05"], dtype=np.float64)
        else:
            raise ValueError("unsupported coefficient center_source")
        source_fingerprint = str(stats["stats_fingerprint"])
    if maximum.shape != (basis.synergy_dim,) or center.shape != (basis.synergy_dim,):
        raise ValueError("coefficient maximum/center must match synergy rank")
    if not np.all(np.isfinite(maximum)) or np.any(maximum <= 0.0):
        raise ValueError("coefficient maximum must contain finite positive values")
    if not np.all(np.isfinite(center)) or np.any(center < 0.0) or np.any(center >= maximum):
        raise ValueError("coefficient center must lie in [0, maximum)")
    raw_temperature = _cfg_get(config, "temperature", 1.0)
    temperature = np.asarray(raw_temperature, dtype=np.float64)
    if temperature.ndim == 0:
        temperature = np.full(basis.synergy_dim, float(temperature), dtype=np.float64)
    if temperature.shape != (basis.synergy_dim,) or not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("coefficient temperature must be positive scalar or per-dimension vector")
    payload = {
        "kind": kind,
        "maximum": maximum.tolist(),
        "center": center.tolist(),
        "temperature": temperature.tolist(),
        "source_fingerprint": source_fingerprint,
        "basis_fingerprint": formal_basis_fingerprint,
    }
    return CoefficientTransform(
        kind=kind,
        maximum=maximum.astype(np.float32),
        center=center.astype(np.float32),
        temperature=temperature.astype(np.float32),
        fingerprint=_json_sha256(payload),
        source_fingerprint=source_fingerprint,
    )


def _load_tonic_baseline(
    config: Any,
    *,
    action_dim: int,
    actuator_names: tuple[str, ...],
    excitation_bounds: np.ndarray,
    basis_fingerprint: str,
) -> tuple[np.ndarray, str]:
    if bool(_cfg_get(config, "learned_full_dimensional", False)):
        raise ValueError("early-synergy tonic baseline must be fixed, never learned full-dimensional")
    kind = str(_cfg_get(config, "kind", "zero"))
    if kind == "zero":
        values = np.zeros(action_dim, dtype=np.float64)
        source = {"kind": "zero", "basis_fingerprint": basis_fingerprint}
    elif kind == "fixed_artifact":
        source_path = Path(str(_cfg_get(config, "path", "")))
        if not source_path.is_file():
            raise FileNotFoundError(f"tonic baseline artifact does not exist: {source_path}")
        with np.load(source_path, allow_pickle=False) as data:
            if set(data.files) != {"baseline", "actuator_names", "basis_fingerprint", "fingerprint"}:
                raise ValueError("tonic baseline artifact fields differ from contract")
            values = np.asarray(data["baseline"], dtype=np.float64)
            names = tuple(str(value) for value in np.asarray(data["actuator_names"]).tolist())
            bound_basis = _scalar_string(data["basis_fingerprint"])
            supplied = _scalar_string(data["fingerprint"])
        if names != actuator_names or bound_basis != basis_fingerprint:
            raise ValueError("tonic baseline artifact schema/basis binding mismatch")
        source = {
            "kind": kind,
            "values": values.tolist(),
            "actuator_names": list(names),
            "basis_fingerprint": bound_basis,
        }
        if supplied != _json_sha256(source):
            raise ValueError("tonic baseline artifact fingerprint mismatch")
    else:
        raise ValueError("tonic baseline kind must be zero or fixed_artifact")
    bounds = np.asarray(excitation_bounds, dtype=np.float64)
    if values.shape != (action_dim,) or not np.all(np.isfinite(values)):
        raise ValueError("tonic baseline must be a finite body-action vector")
    if np.any(values < bounds[:, 0]) or np.any(values > bounds[:, 1]):
        raise ValueError("tonic baseline lies outside physical excitation bounds")
    fingerprint = _json_sha256(source)
    expected = _cfg_get(config, "expected_fingerprint", None)
    if expected not in (None, "") and str(expected) != fingerprint:
        raise ValueError("tonic baseline expected fingerprint mismatch")
    return values.astype(np.float32), fingerprint


def _load_coverage_gate_binding(
    config: Any,
    *,
    formal_basis_fingerprint: str,
    coefficient_upper_bounds: np.ndarray,
) -> dict[str, Any] | None:
    required = bool(_cfg_get(config, "require_coverage_gate", False))
    require_producer_bound = bool(_cfg_get(config, "require_producer_bound_coverage", False))
    path = _cfg_get(config, "coverage_gate_path", None)
    if require_producer_bound and not required:
        raise ValueError("require_producer_bound_coverage requires require_coverage_gate=true")
    if not required and not path:
        return None
    if not path:
        raise ValueError("early-synergy production config requires coverage_gate_path")
    from musclemimic.synergy.oracle_coverage import (
        FORMAL_STATIC_PROXY_COVERAGE_SCHEMA_VERSION,
        load_static_proxy_coverage_gate,
    )

    gate = load_static_proxy_coverage_gate(
        path,
        expected_basis_fingerprint=formal_basis_fingerprint,
        require_passed=required,
    )
    producer_binding = gate.get("proxy_binding", {}).get("producer_binding")
    if required and require_producer_bound:
        if gate["schema_version"] != FORMAL_STATIC_PROXY_COVERAGE_SCHEMA_VERSION:
            raise ValueError(
                "producer-bound coverage requires a formal v4 static proxy gate; legacy v3 evidence is insufficient"
            )
        if not isinstance(producer_binding, Mapping):
            raise ValueError("formal v4 static proxy coverage gate requires a valid producer_binding")
        _require_sha256(
            producer_binding.get("producer_manifest_fingerprint"),
            "coverage producer_manifest_fingerprint",
        )
    expected_gate_fingerprint = _cfg_get(config, "expected_coverage_gate_fingerprint", None)
    expected_proxy_fingerprint = _cfg_get(config, "expected_coverage_proxy_fingerprint", None)
    if required and not expected_gate_fingerprint:
        raise ValueError("production early-synergy config requires expected_coverage_gate_fingerprint")
    if required and not expected_proxy_fingerprint:
        raise ValueError("production early-synergy config requires expected_coverage_proxy_fingerprint")
    required_thresholds = _cfg_get(config, "required_coverage_thresholds", None)
    if required and required_thresholds is None:
        raise ValueError("production early-synergy config requires required_coverage_thresholds")
    if required_thresholds is not None:
        configured_thresholds = dict(required_thresholds)
        gate_thresholds = dict(gate["thresholds"])
        if configured_thresholds != gate_thresholds:
            raise ValueError("static proxy coverage thresholds differ from the pinned config")
    if expected_gate_fingerprint and str(expected_gate_fingerprint) != gate["artifact_fingerprint"]:
        raise ValueError("static proxy coverage gate fingerprint differs from config")
    if expected_proxy_fingerprint and str(expected_proxy_fingerprint) != gate["proxy_fingerprint"]:
        raise ValueError("static proxy coverage proxy fingerprint differs from config")
    require_phase_conditioned = bool(_cfg_get(config, "require_phase_conditioned_coverage", False))
    required_phase_schema_fingerprint = _cfg_get(
        config,
        "required_coverage_phase_schema_fingerprint",
        None,
    )
    if require_phase_conditioned and not required_phase_schema_fingerprint:
        raise ValueError("phase-conditioned coverage requires a pinned semantic phase schema fingerprint")
    if required_phase_schema_fingerprint:
        required_phase_schema_fingerprint = _require_sha256(
            str(required_phase_schema_fingerprint),
            "required_coverage_phase_schema_fingerprint",
        )
    gate_phase_schema_fingerprint = gate["proxy_binding"]["phase_schema_fingerprint"]
    if required_phase_schema_fingerprint and (required_phase_schema_fingerprint != gate_phase_schema_fingerprint):
        raise ValueError("static proxy coverage semantic phase schema differs from config")
    minimum_required_phases = int(_cfg_get(config, "min_required_coverage_phases", 1))
    required_phase_ids = tuple(gate["thresholds"]["required_phase_ids"])
    if minimum_required_phases <= 0:
        raise ValueError("min_required_coverage_phases must be positive")
    if require_phase_conditioned and (
        not gate["proxy_binding"]["phase_id_included"] or len(required_phase_ids) < minimum_required_phases
    ):
        raise ValueError("static proxy coverage must bind phase ids and enough explicitly required key phases")
    gate_upper_bounds = np.asarray(
        gate["solver"]["coefficient_upper_bounds"],
        dtype=np.float64,
    )
    runtime_upper_bounds = np.asarray(coefficient_upper_bounds, dtype=np.float64)
    if gate_upper_bounds.shape != runtime_upper_bounds.shape or not np.allclose(
        gate_upper_bounds,
        runtime_upper_bounds,
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError(
            "static proxy coverage coefficient bounds differ from the runtime bounded coefficient transform"
        )
    binding = {
        "schema_version": gate["schema_version"],
        "artifact_fingerprint": gate["artifact_fingerprint"],
        "passed": bool(gate["passed"]),
        "proxy_fingerprint": gate["proxy_fingerprint"],
        "phase_schema_fingerprint": gate_phase_schema_fingerprint,
        "required_phase_ids": list(required_phase_ids),
        "coefficient_upper_bounds": gate_upper_bounds.tolist(),
    }
    if producer_binding is not None:
        binding["producer_binding"] = dict(producer_binding)
        binding["producer_manifest_fingerprint"] = _require_sha256(
            producer_binding.get("producer_manifest_fingerprint"),
            "coverage producer_manifest_fingerprint",
        )
    return binding


def _load_primitive_source_contract(
    config: Any,
    *,
    basis: LoadedSynergyBasis,
    actuator_schema_hash_value: str,
    control_range_hash: str,
    runtime_model_hash: str | None,
) -> dict[str, Any] | None:
    required = bool(_cfg_get(config, "require_primitive_source_contract", False))
    path = _cfg_get(config, "primitive_source_manifest_path", None)
    if not required and not path:
        return None
    if not path:
        raise ValueError("early-synergy production config requires primitive_source_manifest_path")
    expected_fingerprint = _cfg_get(
        config,
        "expected_primitive_source_manifest_fingerprint",
        None,
    )
    if required and not expected_fingerprint:
        raise ValueError("production early-synergy config requires expected_primitive_source_manifest_fingerprint")
    from musclemimic.synergy.primitive_manifest import load_primitive_source_manifest

    source = load_primitive_source_manifest(
        path,
        expected_fingerprint=(None if not expected_fingerprint else str(expected_fingerprint)),
    )
    manifest = source.manifest
    expected_target_skill = str(_cfg_get(config, "expected_target_skill_id", "")).strip()
    if required and not expected_target_skill:
        raise ValueError("production early-synergy config requires expected_target_skill_id")
    if expected_target_skill and manifest["target_skill_id"] != expected_target_skill:
        raise ValueError("primitive source manifest target skill differs from config")
    expected_excluded = _cfg_get(config, "expected_excluded_target_motion_paths", None)
    if required and not expected_excluded:
        raise ValueError("production early-synergy config requires expected_excluded_target_motion_paths")
    if expected_excluded is not None:
        from musclemimic.distill.motion_identity import (
            normalize_relative_motion_path,
            stable_motion_uid,
        )

        configured_inventory = [
            {
                "path": normalize_relative_motion_path(str(path)),
                "motion_uid": stable_motion_uid(str(path)),
            }
            for path in expected_excluded
        ]
        if configured_inventory != manifest["excluded_target_motions"]:
            raise ValueError("primitive source target-motion exclusion inventory differs from config")
    if manifest["actuator_schema_hash"] != actuator_schema_hash_value:
        raise ValueError("primitive source actuator schema differs from runtime body interface")
    if manifest["transform_ctrlrange_schema_hash"] != control_range_hash:
        raise ValueError("primitive source control range differs from runtime body interface")
    if required and not runtime_model_hash:
        raise ValueError("primitive source contract requires runtime MuJoCo model binding")
    runtime_model_compatibility = str(
        _cfg_get(
            config,
            "primitive_runtime_model_compatibility",
            "exact_runtime_model",
        )
    )
    if runtime_model_compatibility not in {
        "exact_runtime_model",
        "portable_body_action_abi",
    }:
        raise ValueError("unsupported primitive runtime model compatibility mode")
    if (
        runtime_model_compatibility == "exact_runtime_model"
        and runtime_model_hash
        and manifest["model_hash"] != runtime_model_hash
    ):
        raise ValueError("primitive source model hash differs from runtime MuJoCo model")
    source_dataset_fingerprint = str(basis.manifest.get("source_dataset_fingerprint", ""))
    if manifest["source_dataset_fingerprint"] != source_dataset_fingerprint:
        raise ValueError("primitive source dataset fingerprint differs from formal synergy basis")
    expected_binding = {
        "schema_version": manifest["schema_version"],
        "manifest_fingerprint": source.fingerprint,
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "primitive_only": True,
        "contains_target_skill_rollouts": False,
        "target_skill_id": manifest["target_skill_id"],
        "excluded_target_motions": list(manifest["excluded_target_motions"]),
        "primitive_task_ids": list(manifest["primitive_task_ids"]),
        "model_hash": manifest["model_hash"],
        "transform_ctrlrange_schema_hash": manifest["transform_ctrlrange_schema_hash"],
    }
    if basis.manifest.get("primitive_source_binding") != expected_binding:
        raise ValueError("formal synergy basis primitive-source binding differs from source manifest")
    return {
        **expected_binding,
        "runtime_model_compatibility": runtime_model_compatibility,
    }


def _validate_excitation_transform(
    basis: LoadedSynergyBasis,
    names: tuple[str, ...],
    *,
    runtime_ctrlrange: np.ndarray | None,
    require_runtime_binding: bool,
) -> str:
    manifest = basis.manifest
    if manifest.get("signal_kind") != EXCITATION_SIGNAL_KIND:
        raise ValueError("early-synergy basis must represent physical unit excitation")
    transform = manifest.get("transform")
    if not isinstance(transform, Mapping):
        raise ValueError("early-synergy basis has no explicit excitation transform")
    required = {
        "kind": UNIT_EXCITATION_TRANSFORM,
        "formula": MUSCLE_EXCITATION_FORMULA,
        "roundoff_policy": MUSCLE_EXCITATION_ROUNDOFF_POLICY,
        "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
    }
    for key, expected in required.items():
        if transform.get(key) != expected:
            raise ValueError(f"early-synergy excitation transform has invalid {key}")
    if transform.get("raw_signal_kind") not in {"applied_ctrl", "teacher_ctrl_physical", "raw_ctrl"}:
        raise ValueError("early-synergy excitation transform has invalid raw signal kind")
    transform_names = tuple(str(name) for name in transform.get("actuator_names", ()))
    if transform_names != names:
        raise ValueError("early-synergy excitation transform schema/order mismatch")
    ctrlrange = validate_unit_muscle_ctrlrange(
        names,
        transform.get("ctrlrange"),
    )
    validate_muscle_channel_contract(
        transform.get("muscle_channel_contract"),
        expected_names=names,
    )
    expected_hash = ctrlrange_schema_hash(names, ctrlrange)
    if transform.get("ctrlrange_schema_hash") != expected_hash:
        raise ValueError("early-synergy excitation control-range hash mismatch")
    if runtime_ctrlrange is None:
        if require_runtime_binding:
            raise ValueError("early-synergy production config requires runtime MuJoCo ctrlrange binding")
        return expected_hash
    runtime = np.asarray(runtime_ctrlrange, dtype=np.float64)
    validate_unit_muscle_ctrlrange(names, runtime)
    if not np.allclose(runtime, ctrlrange, rtol=0.0, atol=1e-12):
        raise ValueError("formal synergy excitation control ranges differ from the runtime MuJoCo model")
    runtime_hash = ctrlrange_schema_hash(names, runtime)
    if runtime_hash != expected_hash:
        raise ValueError("runtime actuator control-range hash differs from formal synergy basis")
    return runtime_hash


def _validate_basis_selection(basis: LoadedSynergyBasis, config: Any) -> None:
    forbid_fallback = bool(_cfg_get(config, "forbid_fallback_selected_basis", True))
    require_all = bool(_cfg_get(config, "require_all_basis_gates", True))
    manifest = basis.manifest
    region = str(manifest.get("region", ""))
    expected_region = str(_cfg_get(config, "expected_basis_region", "") or "").strip()
    if expected_region and region != expected_region:
        raise ValueError(
            "early-synergy basis region differs from the pinned config: "
            f"expected={expected_region!r} actual={region!r}"
        )
    required_thresholds = _cfg_get(config, "required_selection_thresholds", None)
    if required_thresholds is not None:
        required_thresholds = dict(required_thresholds)
    required_hybrid_thresholds = _cfg_get(config, "required_hybrid_thresholds", None)
    if required_hybrid_thresholds is not None:
        required_hybrid_thresholds = dict(required_hybrid_thresholds)
    required_hybrid_dynamic_thresholds = _cfg_get(
        config,
        "required_hybrid_dynamic_thresholds",
        None,
    )
    if required_hybrid_dynamic_thresholds is not None:
        required_hybrid_dynamic_thresholds = dict(required_hybrid_dynamic_thresholds)
    require_hybrid_dynamic_coverage = bool(
        _cfg_get(config, "require_hybrid_dynamic_coverage", False)
    )
    if region != "hybrid_global_regional" and (
        required_hybrid_thresholds is not None
        or required_hybrid_dynamic_thresholds is not None
        or require_hybrid_dynamic_coverage
    ):
        raise ValueError("hybrid-only action gates require a hybrid_global_regional basis")
    reasons: list[str] = []
    if region == "hybrid_global_regional":
        reasons.extend(
            _validate_hybrid_basis_selection(
                basis,
                required_thresholds=required_thresholds,
                required_hybrid_thresholds=required_hybrid_thresholds,
                required_hybrid_dynamic_thresholds=required_hybrid_dynamic_thresholds,
                require_hybrid_dynamic_coverage=require_hybrid_dynamic_coverage,
            )
        )
    elif region == "regional_composite":
        components = manifest.get("component_artifacts")
        if not isinstance(components, Mapping) or not components:
            raise ValueError("regional composite has no component artifact gate bindings")
        for label, descriptor in components.items():
            if not isinstance(descriptor, Mapping):
                raise ValueError("regional composite component descriptor is invalid")
            component = load_fixed_synergy_basis(
                str(descriptor.get("artifact_path", "")),
                expected_actuator_names=None,
            )
            if component.manifest.get("artifact_fingerprint") != descriptor.get("artifact_fingerprint"):
                raise ValueError(f"regional component {label!r} fingerprint mismatch")
            if component.manifest.get("primitive_source_binding") != manifest.get("primitive_source_binding"):
                raise ValueError(f"regional component {label!r} primitive-source binding mismatch")
            reasons.append(
                _validate_formal_selection_manifest(
                    component.manifest,
                    required_thresholds=required_thresholds,
                    selected_basis=component.basis,
                    selected_actuator_names=component.actuator_names,
                )
            )
    else:
        reasons.append(
            _validate_formal_selection_manifest(
                manifest,
                required_thresholds=required_thresholds,
                selected_basis=basis.basis,
                selected_actuator_names=basis.actuator_names,
            )
        )
    if forbid_fallback and any(reason.startswith("fallback_") for reason in reasons):
        raise ValueError("fallback-selected synergy basis is forbidden for hard Stage-1 action restriction")
    if require_all and (not reasons or any(reason != _STRICT_SELECTION_REASON for reason in reasons)):
        raise ValueError("early-synergy basis did not pass every VAF/stability rank-selection gate")


def _validate_hybrid_basis_selection(
    basis: LoadedSynergyBasis,
    *,
    required_thresholds: Mapping[str, Any] | None,
    required_hybrid_thresholds: Mapping[str, Any] | None,
    required_hybrid_dynamic_thresholds: Mapping[str, Any] | None,
    require_hybrid_dynamic_coverage: bool,
) -> list[str]:
    manifest = basis.manifest
    if manifest.get("hybrid_schema_version") != "hybrid_global_regional_basis_v1":
        raise ValueError("hybrid synergy basis has an unsupported schema")
    if manifest.get("artifact_role") != "primary_hybrid_global_regional":
        raise ValueError("production action loader requires a primary hybrid artifact")
    components = manifest.get("source_components")
    if not isinstance(components, Mapping) or set(components) != {"regional", "global"}:
        raise ValueError("hybrid source component bindings are incomplete")
    expected_regions = {"regional": "regional_composite", "global": "whole_body"}
    loaded: dict[str, LoadedSynergyBasis] = {}
    for label, expected_region in expected_regions.items():
        descriptor = components[label]
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "region",
            "artifact_path",
            "artifact_fingerprint",
        }:
            raise ValueError(f"hybrid {label} source descriptor fields differ from contract")
        if descriptor.get("region") != expected_region:
            raise ValueError(f"hybrid {label} source descriptor region mismatch")
        supplied_path = Path(str(descriptor.get("artifact_path", "")))
        if not supplied_path.is_absolute():
            raise ValueError(f"hybrid {label} source path must be absolute")
        source = load_fixed_synergy_basis(
            supplied_path,
            expected_actuator_names=basis.actuator_names,
        )
        if Path(str(source.source_path)).resolve() != supplied_path.resolve():
            raise ValueError(f"hybrid {label} reloaded source path differs from manifest")
        if source.manifest.get("artifact_fingerprint") != descriptor.get("artifact_fingerprint"):
            raise ValueError(f"hybrid {label} source fingerprint mismatch")
        if source.manifest.get("region") != expected_region:
            raise ValueError(f"hybrid {label} reloaded source region mismatch")
        for field in (
            "signal_kind",
            "source_dataset_fingerprint",
            "teacher_checkpoint_fingerprint",
            "primitive_source_binding",
        ):
            if source.manifest.get(field) != manifest.get(field):
                raise ValueError(f"hybrid {label} source {field} differs from primary artifact")
        if _json_sha256(source.manifest.get("split_provenance")) != _json_sha256(
            manifest.get("split_provenance")
        ):
            raise ValueError(f"hybrid {label} source split provenance mismatch")
        validate_decoder_synergy_basis(source)
        loaded[label] = source

    construction = manifest.get("hybrid_construction")
    if not isinstance(construction, Mapping):
        raise ValueError("hybrid construction manifest is absent")
    _validate_required_hybrid_thresholds(
        construction,
        required_thresholds=required_hybrid_thresholds,
    )
    source_fingerprints = manifest.get("source_basis_fingerprints")
    if not isinstance(source_fingerprints, Mapping) or set(source_fingerprints) != {"regional", "global"}:
        raise ValueError("hybrid source basis fingerprints are incomplete")
    for label in ("regional", "global"):
        artifact_fingerprint = loaded[label].manifest.get("artifact_fingerprint")
        if source_fingerprints[label] != artifact_fingerprint:
            raise ValueError(f"hybrid {label} top-level source fingerprint mismatch")
        if construction.get(f"{label}_source_fingerprint") != artifact_fingerprint:
            raise ValueError(f"hybrid {label} construction source fingerprint mismatch")
    if manifest.get("hybrid_matrix_content_sha256") != construction.get("matrix_content_sha256"):
        raise ValueError("hybrid top-level matrix hash differs from construction manifest")

    validate_hybrid_basis_result(
        HybridBasisResult(
            basis=np.asarray(basis.basis, dtype=np.float64),
            muscle_names=basis.actuator_names,
            manifest=dict(construction),
        ),
        regional_basis=loaded["regional"].basis,
        global_basis=loaded["global"].basis,
    )
    _validate_hybrid_dynamic_coverage(
        basis,
        require_coverage=require_hybrid_dynamic_coverage,
        required_thresholds=required_hybrid_dynamic_thresholds,
    )

    reasons = _validate_regional_source_selection(
        loaded["regional"],
        parent_manifest=manifest,
        required_thresholds=required_thresholds,
    )
    reasons.append(
        _validate_formal_selection_manifest(
            loaded["global"].manifest,
            required_thresholds=required_thresholds,
            selected_basis=loaded["global"].basis,
            selected_actuator_names=loaded["global"].actuator_names,
        )
    )
    return reasons


def _validate_regional_source_selection(
    regional: LoadedSynergyBasis,
    *,
    parent_manifest: Mapping[str, Any],
    required_thresholds: Mapping[str, Any] | None,
) -> list[str]:
    components = regional.manifest.get("component_artifacts")
    if not isinstance(components, Mapping) or not components:
        raise ValueError("hybrid regional source has no bound regional components")
    raw_descriptors = regional.manifest.get("composite_regions")
    if not isinstance(raw_descriptors, list) or any(
        not isinstance(item, Mapping) for item in raw_descriptors
    ):
        raise ValueError("hybrid regional source has invalid composite descriptors")
    descriptors = {str(item.get("region", "")): item for item in raw_descriptors}
    if set(descriptors) != set(components) or len(descriptors) != len(raw_descriptors):
        raise ValueError("hybrid regional component inventory differs from composite descriptors")
    reasons: list[str] = []
    for label, descriptor in components.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError("hybrid regional component descriptor is invalid")
        supplied_path = Path(str(descriptor.get("artifact_path", "")))
        if not supplied_path.is_absolute():
            raise ValueError(f"hybrid regional component {label!r} path must be absolute")
        component = load_fixed_synergy_basis(supplied_path, expected_actuator_names=None)
        if Path(str(component.source_path)).resolve() != supplied_path.resolve():
            raise ValueError(f"hybrid regional component {label!r} source path mismatch")
        if component.manifest.get("artifact_fingerprint") != descriptor.get("artifact_fingerprint"):
            raise ValueError(f"hybrid regional component {label!r} fingerprint mismatch")
        composite_descriptor = descriptors[str(label)]
        if composite_descriptor.get("component_artifact_fingerprint") != component.manifest.get(
            "artifact_fingerprint"
        ):
            raise ValueError(f"hybrid regional component {label!r} composite fingerprint mismatch")
        rows = np.asarray(composite_descriptor.get("row_indices", ()), dtype=np.int64)
        start = int(composite_descriptor.get("column_start", -1))
        stop = int(composite_descriptor.get("column_stop", -1))
        if tuple(str(value) for value in composite_descriptor.get("muscle_names", ())) != component.actuator_names:
            raise ValueError(f"hybrid regional component {label!r} muscle schema mismatch")
        if not np.array_equal(
            np.asarray(regional.basis)[rows, start:stop],
            np.asarray(component.basis),
        ):
            raise ValueError(f"hybrid regional component {label!r} matrix differs from composite block")
        for field in (
            "signal_kind",
            "source_dataset_fingerprint",
            "teacher_checkpoint_fingerprint",
            "primitive_source_binding",
        ):
            if component.manifest.get(field) != parent_manifest.get(field):
                raise ValueError(f"hybrid regional component {label!r} {field} mismatch")
        if _json_sha256(component.manifest.get("split_provenance")) != _json_sha256(
            parent_manifest.get("split_provenance")
        ):
            raise ValueError(f"hybrid regional component {label!r} split provenance mismatch")
        validate_decoder_synergy_basis(component)
        reasons.append(
            _validate_formal_selection_manifest(
                component.manifest,
                required_thresholds=required_thresholds,
                selected_basis=component.basis,
                selected_actuator_names=component.actuator_names,
            )
        )
    return reasons


def _validate_required_hybrid_thresholds(
    construction: Mapping[str, Any],
    *,
    required_thresholds: Mapping[str, Any] | None,
) -> None:
    if required_thresholds is None:
        return
    actual = construction.get("thresholds")
    if not isinstance(actual, Mapping):
        raise ValueError("hybrid construction thresholds are absent")
    if set(required_thresholds) != set(actual):
        raise ValueError("required hybrid threshold fields differ from the construction contract")
    for name, required_value in required_thresholds.items():
        actual_value = actual[name]
        if name == "max_total_rank":
            if _strict_positive_int(required_value, f"required hybrid threshold {name}") != _strict_positive_int(
                actual_value,
                f"hybrid threshold {name}",
            ):
                raise ValueError("hybrid construction thresholds differ from the pinned config")
            continue
        required_number = _positive_finite_float(
            required_value,
            f"required hybrid threshold {name}",
        )
        actual_number = _positive_finite_float(actual_value, f"hybrid threshold {name}")
        if not np.isclose(actual_number, required_number, rtol=0.0, atol=1e-12):
            raise ValueError("hybrid construction thresholds differ from the pinned config")


def _validate_hybrid_dynamic_coverage(
    basis: LoadedSynergyBasis,
    *,
    require_coverage: bool = False,
    required_thresholds: Mapping[str, Any] | None = None,
) -> None:
    manifest = basis.manifest
    binding = manifest.get("hybrid_dynamic_coverage")
    if not isinstance(binding, Mapping) or set(binding) != {
        "requirement",
        "candidate_basis_fingerprint",
        "evidence",
    }:
        raise ValueError("hybrid dynamic coverage binding differs from contract")
    requirement = validate_dynamic_coverage_requirement(binding["requirement"])
    if require_coverage and requirement["required"] is not True:
        raise ValueError("production early-synergy config requires hybrid dynamic coverage")
    if required_thresholds is not None:
        expected_fields = {
            "max_mean_dynamic_gap",
            "max_key_phase_dynamic_gap",
        }
        if set(required_thresholds) != expected_fields:
            raise ValueError("required hybrid dynamic threshold fields differ from contract")
        configured = {
            name: _bounded_finite_float(
                required_thresholds[name],
                f"required hybrid dynamic threshold {name}",
            )
            for name in expected_fields
        }
        if any(
            not np.isclose(
                float(requirement[name]),
                configured[name],
                rtol=0.0,
                atol=1e-12,
            )
            for name in expected_fields
        ):
            raise ValueError("hybrid dynamic thresholds differ from the pinned config")
    expected_candidate = candidate_basis_fingerprint(
        basis.basis,
        muscle_names=basis.actuator_names,
        signal_kind=str(manifest.get("signal_kind", "")),
        region="hybrid_global_regional",
    )
    if binding.get("candidate_basis_fingerprint") != expected_candidate:
        raise ValueError("hybrid dynamic coverage candidate fingerprint mismatch")
    evidence = binding.get("evidence")
    if evidence is None:
        if requirement["required"] is True:
            raise ValueError("hybrid basis lacks required exact rollout coverage evidence")
        return
    validated = validate_dynamic_coverage_gate(
        evidence,
        region="hybrid_global_regional",
        rank=basis.synergy_dim,
        candidate_fingerprint=expected_candidate,
        signal_kind=str(manifest.get("signal_kind", "")),
        max_mean_dynamic_gap=float(requirement["max_mean_dynamic_gap"]),
        max_key_phase_dynamic_gap=float(requirement["max_key_phase_dynamic_gap"]),
        expected_environment_fingerprint=requirement["expected_environment_fingerprint"],
        expected_rollout_manifest_fingerprint=requirement[
            "expected_rollout_manifest_fingerprint"
        ],
    )
    if requirement["required"] is True and validated.get("passed") is not True:
        raise ValueError("hybrid exact rollout coverage gate did not pass")


def _validate_formal_selection_manifest(
    manifest: Mapping[str, Any],
    *,
    required_thresholds: Mapping[str, Any] | None,
    selected_basis: np.ndarray,
    selected_actuator_names: Sequence[str],
) -> str:
    """Recompute rank eligibility instead of trusting a declared reason string."""

    selection = manifest.get("selection")
    selected_metrics = manifest.get("selected_metrics")
    rank_scan = manifest.get("rank_scan")
    if not isinstance(selection, Mapping):
        raise ValueError("formal synergy basis has no rank-selection contract")
    if not isinstance(selected_metrics, Mapping) or not isinstance(rank_scan, Mapping):
        raise ValueError("formal synergy basis lacks selected metrics or rank scan evidence")

    manifest_rank = _strict_positive_int(manifest.get("rank"), "manifest rank")
    selected_rank = _strict_positive_int(
        selection.get("selected_rank"),
        "selection selected_rank",
    )
    if selected_rank != manifest_rank:
        raise ValueError("formal synergy selected rank differs from the saved basis rank")

    raw_thresholds = selection.get("thresholds")
    if not isinstance(raw_thresholds, Mapping):
        raise ValueError("formal synergy selection has no gate thresholds")
    bounded_threshold_names = (
        "min_val_global_vaf",
        "min_val_local_vaf_quantile",
        "local_vaf_quantile",
        "min_initialization_similarity",
        "min_split_half_similarity",
        "min_bootstrap_similarity",
        "min_cross_trial_similarity",
    )
    numerical_threshold_names = (
        "max_basis_condition_number",
        "min_effective_rank_fraction",
    )
    raw_threshold_names = frozenset(raw_thresholds)
    if raw_threshold_names not in {
        frozenset(bounded_threshold_names),
        frozenset((*bounded_threshold_names, *numerical_threshold_names)),
    }:
        raise ValueError("formal synergy gate threshold fields differ from the contract")
    thresholds = {
        name: _bounded_finite_float(raw_thresholds[name], f"selection threshold {name}")
        for name in bounded_threshold_names
    }
    if "max_basis_condition_number" in raw_thresholds:
        thresholds["max_basis_condition_number"] = _positive_finite_float(
            raw_thresholds["max_basis_condition_number"],
            "selection threshold max_basis_condition_number",
        )
        thresholds["min_effective_rank_fraction"] = _bounded_finite_float(
            raw_thresholds["min_effective_rank_fraction"],
            "selection threshold min_effective_rank_fraction",
        )
    if required_thresholds is not None:
        required_names = frozenset(required_thresholds)
        if required_names not in {
            frozenset(bounded_threshold_names),
            frozenset((*bounded_threshold_names, *numerical_threshold_names)),
        }:
            raise ValueError("required selection threshold fields differ from the contract")
        required_values = {
            name: _bounded_finite_float(
                required_thresholds[name],
                f"required selection threshold {name}",
            )
            for name in bounded_threshold_names
        }
        if "max_basis_condition_number" in required_thresholds:
            if "max_basis_condition_number" not in thresholds:
                raise ValueError(
                    "formal synergy numerical thresholds are absent from the saved basis"
                )
            required_values["max_basis_condition_number"] = _positive_finite_float(
                required_thresholds["max_basis_condition_number"],
                "required selection threshold max_basis_condition_number",
            )
            required_values["min_effective_rank_fraction"] = _bounded_finite_float(
                required_thresholds["min_effective_rank_fraction"],
                "required selection threshold min_effective_rank_fraction",
            )
        if any(
            not np.isclose(thresholds[name], value, rtol=0.0, atol=1e-12)
            for name, value in required_values.items()
        ):
            raise ValueError("formal synergy selection thresholds differ from the pinned config")

    dynamic_coverage_gate = _selection_dynamic_coverage_gate(selection)
    expected_selected_candidate: str | None = None
    if dynamic_coverage_gate is not None and dynamic_coverage_gate["required"] is True:
        expected_selected_candidate = candidate_basis_fingerprint(
            selected_basis,
            muscle_names=selected_actuator_names,
            signal_kind=str(manifest.get("signal_kind", "")),
            region=str(manifest.get("region", "")),
        )
    expected_selected_condition = float(np.linalg.cond(np.asarray(selected_basis)))
    expected_selected_effective_rank_fraction = float(
        np.linalg.matrix_rank(np.asarray(selected_basis)) / np.asarray(selected_basis).shape[1]
    )

    computed_eligible: list[int] = []
    reports_by_rank: dict[int, Mapping[str, Any]] = {}
    for raw_rank, raw_report in rank_scan.items():
        if not isinstance(raw_report, Mapping):
            raise ValueError("formal synergy rank scan contains a non-object report")
        try:
            key_rank = int(str(raw_rank))
        except (TypeError, ValueError) as exc:
            raise ValueError("formal synergy rank scan key is not an integer") from exc
        report_rank = _strict_positive_int(raw_report.get("rank"), "rank report rank")
        if key_rank != report_rank or report_rank in reports_by_rank:
            raise ValueError("formal synergy rank scan keys/ranks are inconsistent")
        failures = _selection_gate_failures(
            raw_report,
            thresholds,
            require_primitive_groups=manifest.get("primitive_source_binding") is not None,
            dynamic_coverage_gate=dynamic_coverage_gate,
            signal_kind=str(manifest.get("signal_kind", "")),
            region=str(manifest.get("region", "")),
            expected_candidate_fingerprint=(
                expected_selected_candidate if report_rank == selected_rank else None
            ),
            expected_basis_condition_number=(
                expected_selected_condition if report_rank == selected_rank else None
            ),
            expected_effective_rank_fraction=(
                expected_selected_effective_rank_fraction
                if report_rank == selected_rank
                else None
            ),
        )
        declared_eligible = raw_report.get("eligible")
        if type(declared_eligible) is not bool or declared_eligible != (not failures):
            raise ValueError("formal synergy rank eligibility differs from recomputed gates")
        rejection_reasons = raw_report.get("rejection_reasons")
        if not isinstance(rejection_reasons, list) or any(
            not isinstance(value, str) or not value for value in rejection_reasons
        ):
            raise ValueError("formal synergy rank rejection reasons are invalid")
        if bool(rejection_reasons) != bool(failures):
            raise ValueError("formal synergy rejection evidence differs from recomputed gates")
        reports_by_rank[report_rank] = raw_report
        if not failures:
            computed_eligible.append(report_rank)

    if not reports_by_rank or selected_rank not in reports_by_rank:
        raise ValueError("formal synergy selected rank is absent from rank scan")
    computed_eligible.sort()
    declared_eligible = selection.get("eligible_ranks")
    if (
        not isinstance(declared_eligible, list)
        or any(type(value) is not int or value <= 0 for value in declared_eligible)
        or declared_eligible != computed_eligible
    ):
        raise ValueError("formal synergy eligible rank list differs from recomputed gates")
    if dict(selected_metrics) != dict(reports_by_rank[selected_rank]):
        raise ValueError("formal synergy selected metrics differ from the selected rank scan")

    reason = str(selection.get("reason", ""))
    if computed_eligible:
        if selected_rank != computed_eligible[0] or reason != _STRICT_SELECTION_REASON:
            raise ValueError("formal synergy did not select the smallest gate-eligible rank")
    elif not reason.startswith("fallback_"):
        raise ValueError("formal synergy has no eligible rank and no explicit fallback reason")
    return reason


def _selection_gate_failures(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
    *,
    require_primitive_groups: bool,
    dynamic_coverage_gate: Mapping[str, Any] | None,
    signal_kind: str,
    region: str,
    expected_candidate_fingerprint: str | None,
    expected_basis_condition_number: float | None,
    expected_effective_rank_fraction: float | None,
) -> tuple[str, ...]:
    failures: list[str] = []

    validation = report.get("validation_phase_balanced" if require_primitive_groups else "validation")
    if not isinstance(validation, Mapping):
        raise ValueError("formal synergy rank report lacks validation metrics")
    global_vaf = _finite_float_or_none(validation.get("global_vaf"))
    if global_vaf is None or global_vaf < thresholds["min_val_global_vaf"]:
        failures.append("validation.global_vaf")

    local_quantile = _finite_float_or_none(report.get("validation_local_vaf_quantile"))
    if local_quantile is None or local_quantile < thresholds["min_val_local_vaf_quantile"]:
        failures.append("validation_local_vaf_quantile")
    if "max_basis_condition_number" in thresholds:
        conditioning = report.get("numerical_conditioning")
        if not isinstance(conditioning, Mapping):
            raise ValueError("formal synergy rank report lacks numerical conditioning evidence")
        condition_number = _finite_float_or_none(
            conditioning.get("basis_condition_number")
        )
        effective_rank_fraction = _finite_float_or_none(
            conditioning.get("effective_rank_fraction")
        )
        if condition_number is None or condition_number <= 0.0:
            failures.append("basis_condition_number")
        elif condition_number > thresholds["max_basis_condition_number"]:
            failures.append("basis_condition_number")
        if (
            effective_rank_fraction is None
            or not 0.0 <= effective_rank_fraction <= 1.0
            or effective_rank_fraction < thresholds["min_effective_rank_fraction"]
        ):
            failures.append("effective_rank_fraction")
        if expected_basis_condition_number is not None and (
            condition_number is None
            or not np.isclose(
                condition_number,
                expected_basis_condition_number,
                rtol=1e-6,
                atol=1e-9,
            )
        ):
            raise ValueError(
                "formal selected basis condition number differs from rank evidence"
            )
        if expected_effective_rank_fraction is not None and (
            effective_rank_fraction is None
            or not np.isclose(
                effective_rank_fraction,
                expected_effective_rank_fraction,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "formal selected basis effective rank differs from rank evidence"
            )
    primitive_groups = report.get("primitive_group_validation")
    if require_primitive_groups:
        if not isinstance(primitive_groups, Mapping):
            raise ValueError("formal primitive basis lacks per-task/phase validation evidence")
        collected: list[float] = []
        for group_name in (
            "per_task",
            "per_task_phase",
            "per_trial",
            "per_task_phase_trial",
        ):
            group = primitive_groups.get(group_name)
            if not isinstance(group, Mapping) or not group:
                raise ValueError("formal primitive group validation evidence is incomplete")
            for metrics in group.values():
                if not isinstance(metrics, Mapping):
                    raise ValueError("formal primitive group validation metric is invalid")
                value = _finite_float_or_none(metrics.get("global_vaf"))
                if value is None:
                    raise ValueError("formal primitive group global VAF is invalid")
                collected.append(value)
        declared_minimum = _finite_float_or_none(primitive_groups.get("minimum_global_vaf"))
        actual_minimum = min(collected)
        if declared_minimum is None or not np.isclose(
            declared_minimum,
            actual_minimum,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("formal primitive minimum group VAF is inconsistent")
        if actual_minimum < thresholds["min_val_local_vaf_quantile"]:
            failures.append("primitive_task_phase_global_vaf")
    quantile_level = _finite_float_or_none(report.get("validation_local_vaf_quantile_level"))
    if quantile_level is None or not np.isclose(
        quantile_level,
        thresholds["local_vaf_quantile"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("formal synergy local-VAF quantile level differs from selection threshold")

    for report_name, threshold_name in (
        ("initialization_stability", "min_initialization_similarity"),
        ("split_half_stability", "min_split_half_similarity"),
        ("bootstrap_stability", "min_bootstrap_similarity"),
    ):
        stability = report.get(report_name)
        if not isinstance(stability, Mapping):
            raise ValueError(f"formal synergy rank report lacks {report_name}")
        similarity = _finite_float_or_none(stability.get("mean_similarity"))
        if similarity is None or similarity < thresholds[threshold_name]:
            failures.append(report_name)

    cross_trial = report.get("cross_trial_stability")
    if not isinstance(cross_trial, Mapping) or type(cross_trial.get("available")) is not bool:
        raise ValueError("formal synergy rank report lacks cross-trial availability evidence")
    cross_similarity = _finite_float_or_none(cross_trial.get("mean_similarity"))
    per_task = cross_trial.get("per_task")
    per_task_failed = False
    if per_task is not None:
        if not isinstance(per_task, Mapping) or not per_task:
            raise ValueError("formal synergy cross-trial per-task evidence is invalid")
        for task, task_report in per_task.items():
            if not isinstance(task, str) or not task or not isinstance(task_report, Mapping):
                raise ValueError("formal synergy cross-trial task report is invalid")
            task_mean = _finite_float_or_none(task_report.get("mean_similarity"))
            task_minimum = _finite_float_or_none(task_report.get("min_similarity"))
            if (
                task_mean is None
                or task_minimum is None
                or task_mean < thresholds["min_cross_trial_similarity"]
                or task_minimum < thresholds["min_cross_trial_similarity"]
            ):
                per_task_failed = True
    if (
        cross_trial["available"] is not True
        or cross_similarity is None
        or cross_similarity < thresholds["min_cross_trial_similarity"]
        or per_task_failed
    ):
        failures.append("cross_trial_stability")
    failures.extend(
        _dynamic_coverage_gate_failures(
            report,
            gate=dynamic_coverage_gate,
            signal_kind=signal_kind,
            region=region,
            expected_candidate_fingerprint=expected_candidate_fingerprint,
        )
    )
    return tuple(failures)


def _selection_dynamic_coverage_gate(
    selection: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate the optional rollout gate without changing the offline threshold ABI."""

    raw_gate = selection.get("dynamic_coverage_gate")
    if raw_gate is None:
        return None
    try:
        return validate_dynamic_coverage_requirement(raw_gate)
    except (TypeError, ValueError) as exc:
        raise ValueError("formal synergy dynamic-coverage requirement is invalid") from exc


def _dynamic_coverage_gate_failures(
    report: Mapping[str, Any],
    *,
    gate: Mapping[str, Any] | None,
    signal_kind: str,
    region: str,
    expected_candidate_fingerprint: str | None,
) -> tuple[str, ...]:
    """Revalidate required rollout evidence and classify every failure closed."""

    if gate is None or gate["required"] is not True:
        return ()
    if report.get("dynamic_coverage_required") is not True:
        return ("required_dynamic_coverage_evidence_invalid",)

    evidence = report.get("dynamic_coverage")
    validation_error = report.get("dynamic_coverage_validation_error")
    if evidence is None:
        if validation_error is None:
            return ("required_dynamic_coverage_evidence_missing",)
        if not isinstance(validation_error, str) or not validation_error:
            return ("required_dynamic_coverage_evidence_invalid",)
        return ("required_dynamic_coverage_evidence_invalid",)
    if not isinstance(evidence, Mapping) or validation_error is not None:
        return ("required_dynamic_coverage_evidence_invalid",)

    candidate_fingerprint = report.get("candidate_basis_fingerprint")
    if (
        expected_candidate_fingerprint is not None
        and candidate_fingerprint != expected_candidate_fingerprint
    ):
        return ("required_dynamic_coverage_evidence_invalid",)
    try:
        validated = validate_dynamic_coverage_gate(
            evidence,
            region=region,
            rank=_strict_positive_int(report.get("rank"), "rank report rank"),
            candidate_fingerprint=str(candidate_fingerprint or ""),
            signal_kind=signal_kind,
            max_mean_dynamic_gap=float(gate["max_mean_dynamic_gap"]),
            max_key_phase_dynamic_gap=float(gate["max_key_phase_dynamic_gap"]),
            expected_environment_fingerprint=str(
                gate["expected_environment_fingerprint"]
            ),
            expected_rollout_manifest_fingerprint=str(
                gate["expected_rollout_manifest_fingerprint"]
            ),
        )
    except (TypeError, ValueError):
        return ("required_dynamic_coverage_evidence_invalid",)
    if validated.get("passed") is not True:
        return ("required_dynamic_coverage_gate_failed",)
    return ()


def _strict_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_float_or_none(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _bounded_finite_float(value: Any, label: str) -> float:
    result = _finite_float_or_none(value)
    if result is None or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be finite and lie in [0,1]")
    return result


def _positive_finite_float(value: Any, label: str) -> float:
    result = _finite_float_or_none(value)
    if result is None or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _basis_geometry(basis: np.ndarray) -> tuple[float, float]:
    singular = np.linalg.svd(np.asarray(basis, dtype=np.float64), compute_uv=False)
    if singular.size == 0 or singular[0] <= 0.0:
        raise ValueError("early-synergy basis has no effective directions")
    tolerance = np.finfo(np.float64).eps * max(np.asarray(basis).shape) * singular[0]
    positive = singular[singular > tolerance]
    if positive.size != min(np.asarray(basis).shape):
        condition_number = float("inf")
    else:
        condition_number = float(positive[0] / positive[-1])
    probabilities = singular / np.sum(singular)
    entropy = -float(np.sum(np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0)))
    return condition_number, float(np.exp(entropy))


def _validate_residual_matrix(
    basis: np.ndarray,
    actuator_names: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    matrix = np.asarray(basis, dtype=np.float64)
    names = tuple(str(name) for name in actuator_names)
    if matrix.ndim != 2 or min(matrix.shape) <= 0 or matrix.shape[0] != len(names):
        raise ValueError("structured residual basis must be [body_action_dim,residual_dim]")
    if not np.all(np.isfinite(matrix)) or np.any(np.linalg.norm(matrix, axis=0) <= 1e-12):
        raise ValueError("structured residual basis must contain finite non-empty columns")
    if not names or len(set(names)) != len(names):
        raise ValueError("structured residual actuator names must be unique and match rows")
    return matrix, names


def residual_matrix_fingerprint(
    basis: np.ndarray,
    actuator_names: Sequence[str],
) -> str:
    """Hash the exact normalized float32 residual matrix consumed at runtime."""

    matrix, names = _validate_residual_matrix(basis, actuator_names)
    canonical = np.ascontiguousarray(matrix, dtype="<f4")
    header = {
        "schema_version": "early_synergy_residual_matrix_content_v1",
        "shape": list(canonical.shape),
        "dtype": "little_endian_float32",
        "actuator_names": list(names),
        "actuator_schema_hash": actuator_schema_hash(names),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _validate_residual_fit_contract(
    value: Mapping[str, Any],
    *,
    source_basis_fingerprint: str,
    actuator_names: Sequence[str],
    allowed_muscle_mask: np.ndarray,
    basis: np.ndarray,
) -> dict[str, Any]:
    """Validate the train-only, phase-structured residual derivation evidence."""

    if not isinstance(value, Mapping):
        raise ValueError("structured residual fit_contract must be an object")
    required = {
        "schema_version",
        "derivation",
        "fit_scope",
        "passed",
        "source_basis_fingerprint",
        "source_dataset_fingerprint",
        "primitive_source_manifest_fingerprint",
        "coefficient_statistics_fingerprint",
        "coefficient_upper_bounds",
        "reference_alpha",
        "train_source_fingerprint",
        "validation_source_fingerprint",
        "residual_matrix_fingerprint",
        "mask_contract",
        "groups",
        "sample_balancing",
        "projection_solver",
        "projection_solver_parameters",
        "metrics",
        "thresholds",
        "fit_contract_fingerprint",
    }
    if set(value) != required:
        raise ValueError(f"structured residual fit_contract fields differ from contract: {sorted(value)}")
    if value.get("schema_version") != "early_synergy_residual_fit_v1":
        raise ValueError("unsupported structured residual fit contract")
    if value.get("derivation") != "phase_grouped_unexplained_excitation_svd_v1":
        raise ValueError("structured residual derivation is not the supported Phase-A method")
    if value.get("fit_scope") != "train_only_validation_held_out":
        raise ValueError("structured residual fit must be train-only with held-out validation")
    if type(value.get("passed")) is not bool or not value["passed"]:
        raise ValueError("structured residual fit contract did not pass its held-out gates")
    expected_source = _require_sha256(
        source_basis_fingerprint,
        "source_basis_fingerprint",
    )
    if value.get("source_basis_fingerprint") != expected_source:
        raise ValueError("structured residual fit contract belongs to another primary basis")
    for field in (
        "source_dataset_fingerprint",
        "primitive_source_manifest_fingerprint",
        "coefficient_statistics_fingerprint",
        "train_source_fingerprint",
        "validation_source_fingerprint",
        "residual_matrix_fingerprint",
    ):
        _require_sha256(value.get(field), f"structured residual {field}")
    names = tuple(str(name) for name in actuator_names)
    matrix = np.asarray(basis, dtype=np.float64)
    if value.get("residual_matrix_fingerprint") != residual_matrix_fingerprint(
        matrix,
        names,
    ):
        raise ValueError("structured residual fit contract matrix fingerprint differs from runtime R")
    upper = np.asarray(value.get("coefficient_upper_bounds"), dtype=np.float64)
    if upper.ndim != 1 or upper.size <= 0 or not np.all(np.isfinite(upper)) or np.any(upper <= 0.0):
        raise ValueError("structured residual coefficient upper bounds are invalid")
    alpha = float(value.get("reference_alpha", float("nan")))
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("structured residual reference_alpha must lie in (0,1]")

    mask_contract = value.get("mask_contract")
    if not isinstance(mask_contract, Mapping):
        raise ValueError("structured residual mask_contract must be an object")
    mask_required = {
        "schema_version",
        "fingerprint",
        "actuator_schema_hash",
        "groups",
        "union_allowed_muscle_names",
        "union_allowed_muscle_mask",
    }
    if set(mask_contract) != mask_required:
        raise ValueError("structured residual mask_contract fields are invalid")
    if mask_contract.get("schema_version") != "early_synergy_residual_mask_v1":
        raise ValueError("unsupported structured residual mask contract")
    _require_sha256(mask_contract.get("fingerprint"), "residual mask fingerprint")
    if mask_contract.get("actuator_schema_hash") != actuator_schema_hash(names):
        raise ValueError("structured residual mask actuator schema differs from basis")
    union_mask = np.asarray(mask_contract.get("union_allowed_muscle_mask"))
    if (
        union_mask.dtype != np.bool_
        or union_mask.shape != (len(names),)
        or not np.array_equal(union_mask, allowed_muscle_mask)
    ):
        raise ValueError("structured residual union allowed-muscle mask is inconsistent")
    union_names = [names[index] for index in np.flatnonzero(union_mask).tolist()]
    if mask_contract.get("union_allowed_muscle_names") != union_names:
        raise ValueError("structured residual union allowed-muscle names are inconsistent")
    unsigned_mask = {key: item for key, item in mask_contract.items() if key != "fingerprint"}
    if mask_contract.get("fingerprint") != _json_sha256(unsigned_mask):
        raise ValueError("structured residual mask fingerprint mismatch")

    groups = value.get("groups")
    mask_groups = mask_contract.get("groups")
    if (
        not isinstance(groups, list)
        or not groups
        or not isinstance(mask_groups, list)
        or len(groups) != len(mask_groups)
    ):
        raise ValueError("structured residual group descriptors are missing or inconsistent")
    expected_start = 0
    seen_names: set[str] = set()
    seen_task_phases: set[tuple[str, int]] = set()
    for group, mask_group in zip(groups, mask_groups, strict=True):
        if not isinstance(group, Mapping) or set(group) != {
            "name",
            "task_phase_selectors",
            "allowed_muscle_names",
            "allowed_row_indices",
            "rank",
            "column_start",
            "column_stop",
            "train_sample_count",
            "weighted_singular_values",
            "weighted_directional_energy_fraction",
        }:
            raise ValueError("structured residual group descriptor is invalid")
        mask_fields = {
            "name",
            "task_phase_selectors",
            "allowed_muscle_names",
            "allowed_row_indices",
            "rank",
        }
        if (
            not isinstance(mask_group, Mapping)
            or set(mask_group) != mask_fields
            or {field: group[field] for field in mask_fields} != dict(mask_group)
        ):
            raise ValueError("structured residual fitted group differs from mask contract")
        group_name = group.get("name")
        if not isinstance(group_name, str) or not group_name or group_name in seen_names:
            raise ValueError("structured residual group names must be unique and non-empty")
        seen_names.add(group_name)
        selectors = group.get("task_phase_selectors")
        if not isinstance(selectors, Mapping) or not selectors:
            raise ValueError("structured residual group requires task-phase selectors")
        for task, phases in selectors.items():
            if not isinstance(task, str) or not task or not isinstance(phases, list) or not phases:
                raise ValueError("structured residual task-phase selector is invalid")
            if phases != sorted(phases) or any(type(phase) is not int or phase < 0 for phase in phases):
                raise ValueError("structured residual phase ids must be sorted non-negative integers")
            for phase in phases:
                key = (task, phase)
                if key in seen_task_phases:
                    raise ValueError("structured residual task-phase selectors overlap")
                seen_task_phases.add(key)
        rows = group.get("allowed_row_indices")
        muscle_names = group.get("allowed_muscle_names")
        if not isinstance(rows, list) or not rows or rows != sorted(rows):
            raise ValueError("structured residual group allowed rows are invalid")
        if any(type(index) is not int or index < 0 or index >= len(names) for index in rows):
            raise ValueError("structured residual group allowed row lies outside body schema")
        if len(rows) != len(set(rows)) or muscle_names != [names[index] for index in rows]:
            raise ValueError("structured residual group muscle names/order differ from allowed rows")
        rank = group.get("rank")
        start = group.get("column_start")
        stop = group.get("column_stop")
        if (
            type(rank) is not int
            or rank <= 0
            or type(start) is not int
            or type(stop) is not int
            or start != expected_start
            or stop != start + rank
        ):
            raise ValueError("structured residual group column slices are invalid")
        train_sample_count = group.get("train_sample_count")
        singular_values = np.asarray(
            group.get("weighted_singular_values"),
            dtype=np.float64,
        )
        directional_fraction = float(group.get("weighted_directional_energy_fraction", float("nan")))
        if type(train_sample_count) is not int or train_sample_count < rank:
            raise ValueError("structured residual group has insufficient train samples")
        if (
            singular_values.ndim != 1
            or singular_values.size < rank
            or not np.all(np.isfinite(singular_values))
            or np.any(singular_values < 0.0)
            or not np.isfinite(directional_fraction)
            or not 0.0 <= directional_fraction <= 1.0
        ):
            raise ValueError("structured residual group SVD evidence is invalid")
        outside = np.ones(len(names), dtype=bool)
        outside[np.asarray(rows, dtype=np.int64)] = False
        if np.any(np.abs(matrix[outside, start:stop]) > 1e-12):
            raise ValueError("structured residual group has support outside its allowed muscles")
        expected_start = stop
    if expected_start != matrix.shape[1]:
        raise ValueError("structured residual groups do not cover all residual columns")

    if value.get("sample_balancing") != ("equal_task_phase_then_trial_mean_quality_then_frame_quality"):
        raise ValueError("structured residual sample-balancing contract is unsupported")
    if value.get("projection_solver") != "scipy_lsq_linear_bounded_exact":
        raise ValueError("structured residual projection solver is unsupported")
    solver_parameters = value.get("projection_solver_parameters")
    if not isinstance(solver_parameters, Mapping) or set(solver_parameters) != {
        "solver_tolerance",
        "solver_max_iterations",
        "energy_epsilon",
    }:
        raise ValueError("structured residual projection solver parameters are invalid")
    solver_tolerance = float(solver_parameters.get("solver_tolerance", float("nan")))
    energy_epsilon = float(solver_parameters.get("energy_epsilon", float("nan")))
    solver_max_iterations = solver_parameters.get("solver_max_iterations")
    if (
        not np.isfinite(solver_tolerance)
        or solver_tolerance <= 0.0
        or type(solver_max_iterations) is not int
        or solver_max_iterations <= 0
        or not np.isfinite(energy_epsilon)
        or energy_epsilon <= 0.0
    ):
        raise ValueError("structured residual projection solver parameters must be positive")
    thresholds = value.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != {
        "min_validation_residual_energy_reduction",
        "min_group_validation_residual_energy_reduction",
        "max_validation_coordinate_saturation_fraction",
    }:
        raise ValueError("structured residual validation thresholds are invalid")
    min_global = _bounded_finite_float(
        thresholds.get("min_validation_residual_energy_reduction"),
        "min_validation_residual_energy_reduction",
    )
    min_group = _bounded_finite_float(
        thresholds.get("min_group_validation_residual_energy_reduction"),
        "min_group_validation_residual_energy_reduction",
    )
    max_saturation = _bounded_finite_float(
        thresholds.get("max_validation_coordinate_saturation_fraction"),
        "max_validation_coordinate_saturation_fraction",
    )
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "train",
        "validation",
        "per_validation_group",
    }:
        raise ValueError("structured residual held-out metrics are invalid")
    for split in ("train", "validation"):
        _validate_residual_reconstruction_metrics(metrics.get(split), label=split)
    per_group = metrics.get("per_validation_group")
    if not isinstance(per_group, Mapping) or set(per_group) != seen_names:
        raise ValueError("structured residual per-group validation metrics are incomplete")
    for group_name, group_metrics in per_group.items():
        _validate_residual_reconstruction_metrics(
            group_metrics,
            label=f"per_validation_group[{group_name!r}]",
        )
    validation = metrics["validation"]
    computed_pass = (
        float(validation["residual_energy_reduction"]) >= min_global
        and float(validation["coordinate_saturation_fraction"]) <= max_saturation
        and all(
            float(group_metrics["residual_energy_reduction"]) >= min_group
            and float(group_metrics["coordinate_saturation_fraction"]) <= max_saturation
            for group_metrics in per_group.values()
        )
    )
    if computed_pass is not value["passed"]:
        raise ValueError("structured residual held-out pass flag differs from metrics")
    supplied_fingerprint = _require_sha256(
        value.get("fit_contract_fingerprint"),
        "structured residual fit_contract_fingerprint",
    )
    unsigned = {key: item for key, item in value.items() if key != "fit_contract_fingerprint"}
    if supplied_fingerprint != _json_sha256(unsigned):
        raise ValueError("structured residual fit contract fingerprint mismatch")
    return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))


def _validate_residual_reconstruction_metrics(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "sample_count",
        "primary_residual_energy",
        "augmented_residual_energy",
        "residual_energy_reduction",
        "coordinate_saturation_fraction",
    }:
        raise ValueError(f"structured residual {label} metrics fields are invalid")
    if type(value.get("sample_count")) is not int or value["sample_count"] <= 0:
        raise ValueError(f"structured residual {label} sample count must be positive")
    for field in ("primary_residual_energy", "augmented_residual_energy"):
        number = float(value.get(field, float("nan")))
        if not np.isfinite(number) or number < 0.0:
            raise ValueError(f"structured residual {label} {field} is invalid")
    _bounded_finite_float(
        value.get("residual_energy_reduction"),
        f"structured residual {label} residual_energy_reduction",
    )
    _bounded_finite_float(
        value.get("coordinate_saturation_fraction"),
        f"structured residual {label} coordinate_saturation_fraction",
    )


def _coefficient_stats_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            canonical[key] = np.asarray(value, dtype=np.float64).tolist()
        elif isinstance(value, np.generic):
            canonical[key] = value.item()
        else:
            canonical[key] = value
    return _json_sha256(canonical)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    result = str(value)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{label} must be lowercase 64-hex")
    return result


def _scalar_string(value: Any) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("artifact scalar string field has a non-scalar shape")
    return str(array.item())


def _scalar_int(value: Any, label: str) -> int:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"artifact {label} field has a non-scalar shape")
    try:
        return int(array.item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"artifact {label} field is not an integer") from exc
