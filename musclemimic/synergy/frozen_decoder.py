"""Portable, content-bound body-synergy decoder shared by every stage.

The neural policy is allowed to predict only the unconstrained ``c`` and
``rho`` coordinates.  This module owns the sole executable definition of the
fixed decoder that turns those coordinates into the normalized 354-D body
action ABI::

    c = maximum * sigmoid(raw_c / temperature + logit(center / maximum))
    rho = alpha * tanh(raw_rho)
    excitation = clip(tonic + W c + R rho, lower, upper)

The numerical arrays are stored next to a strict :class:`BodySynergyContractV2`
manifest.  Both the array file and the decoded semantic core are fingerprinted,
so modifying W, the coefficient transform, tonic, R, bounds, alpha, or ordered
actuator names fails closed at load time.
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
from musclemimic.synergy.multistage_contract import (
    FIXED_SYNERGY_MODE,
    FIXED_SYNERGY_RESIDUAL_MODE,
    BodySynergyContractV2,
    canonical_json_sha256,
    load_body_synergy_contract,
)
from musclemimic.synergy.schema import ctrlrange_schema_hash

FROZEN_BODY_DECODER_SCHEMA_VERSION = "frozen_body_synergy_decoder_v2"
FROZEN_BODY_DECODER_EXECUTION_BINDING_SCHEMA_VERSION = "frozen_body_decoder_execution_binding_v1"
FROZEN_BODY_DECODER_EXECUTION_BINDING_FIELD = "frozen_body_decoder_execution_binding"
FROZEN_BODY_DECODER_ARRAYS = "frozen_body_decoder.npz"
FROZEN_BODY_DECODER_MANIFEST = "frozen_body_decoder.json"
BODY_SYNERGY_CONTRACT_FILENAME = "body_synergy_contract.json"


class FrozenBodyDecoderJaxParams(NamedTuple):
    """JAX pytree containing every non-trainable decoder value."""

    basis: Any
    excitation_bounds: Any
    coefficient_maximum: Any
    coefficient_bias: Any
    coefficient_temperature: Any
    tonic_baseline: Any
    residual_basis: Any
    residual_alpha: Any


class FrozenBodyDecoderOutput(NamedTuple):
    """Complete output of the shared frozen decoder."""

    body_action: Any
    physical_excitation: Any
    preclip_excitation: Any
    synergy_excitation: Any
    synergy_coefficients: Any
    residual_coefficients: Any
    residual_excitation: Any

    @property
    def effective_excitation(self) -> Any:
        """Explicit name for the verified MuJoCo muscle command.

        ``physical_excitation`` remains the serialized field name for API
        continuity, but under the v2 artifact schema it has exactly these
        effective-excitation semantics.
        """

        return self.physical_excitation


def frozen_body_decoder_core_fingerprint(
    *,
    mode: str,
    actuator_names: Sequence[str],
    residual_alpha: float,
    basis: Any,
    excitation_bounds: Any,
    coefficient_maximum: Any,
    coefficient_center: Any,
    coefficient_temperature: Any,
    tonic_baseline: Any,
    residual_basis: Any,
) -> str:
    """Hash the exact float32 values and semantics executed by the decoder.

    This is the existing frozen-decoder numerical-core identity exposed as a
    builder seam.  Keeping one implementation lets the action-interface
    builder bind a :class:`BodySynergyContractV2` to the exact arrays before
    the frozen artifact is serialized.
    """

    names = tuple(str(name) for name in actuator_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("frozen decoder actuator names must be non-empty and unique")
    canonical_mode = str(mode)
    if canonical_mode not in {FIXED_SYNERGY_MODE, FIXED_SYNERGY_RESIDUAL_MODE}:
        raise ValueError("frozen decoder core requires a fixed-synergy mode")
    alpha = float(residual_alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("frozen decoder residual_alpha must be finite and non-negative")

    arrays = {
        "basis": basis,
        "excitation_bounds": excitation_bounds,
        "coefficient_maximum": coefficient_maximum,
        "coefficient_center": coefficient_center,
        "coefficient_temperature": coefficient_temperature,
        "tonic_baseline": tonic_baseline,
        "residual_basis": residual_basis,
    }
    digest = hashlib.sha256()
    for name, value in arrays.items():
        encoded = name.encode("utf-8")
        array = np.asarray(value, dtype="<f4")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"frozen decoder core array {name!r} must be finite")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    digest.update(
        json.dumps(
            {
                "schema_version": FROZEN_BODY_DECODER_SCHEMA_VERSION,
                "mode": canonical_mode,
                "actuator_names": list(names),
                "actuator_schema_hash": actuator_schema_hash(names),
                "residual_alpha": alpha,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def build_frozen_body_decoder_execution_binding(
    *,
    mode: str,
    actuator_names: Sequence[str],
    residual_alpha: float,
    basis: Any,
    excitation_bounds: Any,
    coefficient_maximum: Any,
    coefficient_center: Any,
    coefficient_temperature: Any,
    tonic_baseline: Any,
    residual_basis: Any,
    basis_fingerprint: str,
    runtime_basis_fingerprint: str,
    coefficient_transform_fingerprint: str,
    coefficient_statistics_fingerprint: str,
    tonic_baseline_fingerprint: str,
    residual_basis_fingerprint: str | None,
    residual_fit_contract_fingerprint: str | None,
    residual_allowed_muscle_mask_fingerprint: str | None,
) -> dict[str, Any]:
    """Build the contract-owned binding for the exact executable decoder.

    Source-artifact fingerprints remain valuable provenance, but they do not
    by themselves identify the arrays copied into the portable decoder.  This
    binding snapshots those declarations and binds them to the already-defined
    numerical-core fingerprint.  It is stored inside
    ``BodySynergyContractV2.source_binding`` and therefore participates in the
    portable contract fingerprint.
    """

    names = tuple(str(name) for name in actuator_names)
    basis_array = np.asarray(basis)
    residual_array = np.asarray(residual_basis)
    if basis_array.ndim != 2 or basis_array.shape[0] != len(names):
        raise ValueError("frozen decoder execution binding basis shape is invalid")
    if residual_array.ndim != 2 or residual_array.shape[0] != len(names):
        raise ValueError("frozen decoder execution binding residual basis shape is invalid")
    bounds = np.asarray(excitation_bounds, dtype=np.float64)
    if bounds.shape != (len(names), 2):
        raise ValueError("frozen decoder execution binding excitation bounds are invalid")

    declarations = {
        "mode": str(mode),
        "body_action_dim": len(names),
        "policy_action_dim": int(basis_array.shape[1] + residual_array.shape[1]),
        "actuator_schema_hash": actuator_schema_hash(names),
        "control_range_hash": ctrlrange_schema_hash(names, bounds),
        "basis_fingerprint": _require_sha256(basis_fingerprint, "basis_fingerprint"),
        "runtime_basis_fingerprint": _require_sha256(runtime_basis_fingerprint, "runtime_basis_fingerprint"),
        "basis_rank": int(basis_array.shape[1]),
        "coefficient_transform_fingerprint": _require_sha256(
            coefficient_transform_fingerprint,
            "coefficient_transform_fingerprint",
        ),
        "coefficient_statistics_fingerprint": _require_sha256(
            coefficient_statistics_fingerprint,
            "coefficient_statistics_fingerprint",
        ),
        "tonic_baseline_fingerprint": _require_sha256(tonic_baseline_fingerprint, "tonic_baseline_fingerprint"),
        "residual_basis_fingerprint": _optional_sha256(residual_basis_fingerprint, "residual_basis_fingerprint"),
        "residual_fit_contract_fingerprint": _optional_sha256(
            residual_fit_contract_fingerprint,
            "residual_fit_contract_fingerprint",
        ),
        "residual_allowed_muscle_mask_fingerprint": _optional_sha256(
            residual_allowed_muscle_mask_fingerprint,
            "residual_allowed_muscle_mask_fingerprint",
        ),
        "residual_dim": int(residual_array.shape[1]),
        "residual_alpha": float(residual_alpha),
    }
    unsigned = {
        "schema_version": FROZEN_BODY_DECODER_EXECUTION_BINDING_SCHEMA_VERSION,
        "decoder_core_fingerprint": frozen_body_decoder_core_fingerprint(
            mode=mode,
            actuator_names=names,
            residual_alpha=residual_alpha,
            basis=basis,
            excitation_bounds=excitation_bounds,
            coefficient_maximum=coefficient_maximum,
            coefficient_center=coefficient_center,
            coefficient_temperature=coefficient_temperature,
            tonic_baseline=tonic_baseline,
            residual_basis=residual_basis,
        ),
        "contract_decoder_declarations": declarations,
    }
    return {
        **unsigned,
        "binding_fingerprint": canonical_json_sha256(unsigned),
    }


def normalized_to_physical(action: Any, excitation_bounds: Any) -> Any:
    """Map normalized actions to the explicitly declared physical range."""

    value = jnp.asarray(action)
    bounds = jnp.asarray(excitation_bounds, dtype=value.dtype)
    if bounds.shape != (value.shape[-1], 2):
        raise ValueError(f"excitation bounds must have shape ({value.shape[-1]}, 2), got {bounds.shape}")
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return lower + 0.5 * (jnp.clip(value, -1.0, 1.0) + 1.0) * (upper - lower)


def physical_to_normalized(excitation: Any, excitation_bounds: Any) -> Any:
    """Map physical excitation to the symmetric body-action ABI."""

    value = jnp.asarray(excitation)
    bounds = jnp.asarray(excitation_bounds, dtype=value.dtype)
    if bounds.shape != (value.shape[-1], 2):
        raise ValueError(f"excitation bounds must have shape ({value.shape[-1]}, 2), got {bounds.shape}")
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return jnp.clip(2.0 * (value - lower) / (upper - lower) - 1.0, -1.0, 1.0)


def decode_frozen_body_action(
    raw_policy_action: Any,
    decoder: FrozenBodyDecoderJaxParams,
) -> FrozenBodyDecoderOutput:
    """Execute the unique pure-JAX fixed-W/tonic/R decoder.

    No host conversion or trainable parameter is used here, so this function
    can be called under ``jit``, ``grad``, and ``vmap`` by both early PPO and
    latent distillation/runtime.
    """

    raw = jnp.asarray(raw_policy_action)
    basis = jnp.asarray(decoder.basis, dtype=raw.dtype)
    residual_basis = jnp.asarray(decoder.residual_basis, dtype=raw.dtype)
    synergy_dim = int(basis.shape[1])
    residual_dim = int(residual_basis.shape[1])
    expected_dim = synergy_dim + residual_dim
    if raw.ndim == 0 or int(raw.shape[-1]) != expected_dim:
        raise ValueError(f"frozen body decoder raw action last dimension must be {expected_dim}, got {raw.shape}")

    raw_coefficients = raw[..., :synergy_dim]
    maximum = jnp.asarray(decoder.coefficient_maximum, dtype=raw.dtype)
    bias = jnp.asarray(decoder.coefficient_bias, dtype=raw.dtype)
    temperature = jnp.asarray(decoder.coefficient_temperature, dtype=raw.dtype)
    coefficients = bounded_synergy_coefficients(
        raw_coefficients,
        maximum=maximum,
        bias=bias,
        temperature=temperature,
    )

    tonic = jnp.asarray(decoder.tonic_baseline, dtype=raw.dtype)
    synergy_excitation = tonic + jnp.matmul(coefficients, basis.T)
    if residual_dim:
        raw_residual = raw[..., synergy_dim:]
        alpha = jnp.asarray(decoder.residual_alpha, dtype=raw.dtype)
        residual_coefficients = alpha * jnp.tanh(raw_residual)
        residual_excitation = jnp.matmul(residual_coefficients, residual_basis.T)
    else:
        residual_coefficients = jnp.zeros((*raw.shape[:-1], 0), dtype=raw.dtype)
        residual_excitation = jnp.zeros_like(synergy_excitation)

    bounds = jnp.asarray(decoder.excitation_bounds, dtype=raw.dtype)
    preclip_excitation = synergy_excitation + residual_excitation
    physical_excitation = jnp.clip(
        preclip_excitation,
        bounds[:, 0],
        bounds[:, 1],
    )
    return FrozenBodyDecoderOutput(
        body_action=physical_to_normalized(physical_excitation, bounds),
        physical_excitation=physical_excitation,
        preclip_excitation=preclip_excitation,
        synergy_excitation=synergy_excitation,
        synergy_coefficients=coefficients,
        residual_coefficients=residual_coefficients,
        residual_excitation=residual_excitation,
    )


@dataclass(frozen=True)
class FrozenBodyDecoder:
    """Validated, serializable frozen body decoder and portable contract."""

    body_synergy_contract: BodySynergyContractV2
    basis: np.ndarray
    excitation_bounds: np.ndarray
    coefficient_maximum: np.ndarray
    coefficient_center: np.ndarray
    coefficient_temperature: np.ndarray
    tonic_baseline: np.ndarray
    residual_basis: np.ndarray

    def __post_init__(self) -> None:
        contract = self.body_synergy_contract
        if not isinstance(contract, BodySynergyContractV2):
            raise TypeError("body_synergy_contract must be a BodySynergyContractV2")
        if contract.mode not in {FIXED_SYNERGY_MODE, FIXED_SYNERGY_RESIDUAL_MODE}:
            raise ValueError("FrozenBodyDecoder requires a fixed-synergy contract")

        body_dim = int(contract.body_action_dim)
        rank = int(contract.basis_rank)
        residual_dim = int(contract.residual_dim)
        arrays = {
            "basis": _finite_array(self.basis, (body_dim, rank)),
            "excitation_bounds": _finite_array(self.excitation_bounds, (body_dim, 2)),
            "coefficient_maximum": _finite_array(self.coefficient_maximum, (rank,)),
            "coefficient_center": _finite_array(self.coefficient_center, (rank,)),
            "coefficient_temperature": _finite_array(self.coefficient_temperature, (rank,)),
            "tonic_baseline": _finite_array(self.tonic_baseline, (body_dim,)),
            "residual_basis": _finite_array(self.residual_basis, (body_dim, residual_dim)),
        }
        basis = arrays["basis"]
        bounds = arrays["excitation_bounds"]
        maximum = arrays["coefficient_maximum"]
        center = arrays["coefficient_center"]
        temperature = arrays["coefficient_temperature"]
        tonic = arrays["tonic_baseline"]
        if np.any(basis < -1e-7):
            raise ValueError("frozen synergy basis must be non-negative")
        unit_bounds = np.broadcast_to(
            np.asarray([0.0, 1.0], dtype=np.float32),
            bounds.shape,
        )
        if not np.array_equal(bounds, unit_bounds):
            raise ValueError("frozen decoder v2 requires exact [0,1] excitation bounds")
        if np.any(maximum <= 0.0) or np.any(center < 0.0) or np.any(center >= maximum):
            raise ValueError("frozen coefficient center must lie in [0, maximum)")
        if np.any(temperature <= 0.0):
            raise ValueError("frozen coefficient temperature must be positive")
        if np.any(tonic < bounds[:, 0] - 1e-7) or np.any(tonic > bounds[:, 1] + 1e-7):
            raise ValueError("frozen tonic baseline lies outside excitation bounds")
        if contract.policy_action_dim != rank + residual_dim:
            raise ValueError("frozen decoder dimensions disagree with BodySynergyContractV2")
        if contract.mode == FIXED_SYNERGY_MODE and residual_dim != 0:
            raise ValueError("fixed_synergy frozen decoder must have an empty R")
        if contract.mode == FIXED_SYNERGY_RESIDUAL_MODE and residual_dim <= 0:
            raise ValueError("fixed_synergy_residual frozen decoder requires R")

        for name, value in arrays.items():
            object.__setattr__(self, name, np.asarray(value, dtype=np.float32))

    @property
    def actuator_names(self) -> tuple[str, ...]:
        return self.body_synergy_contract.actuator_names

    @property
    def body_action_dim(self) -> int:
        return int(self.basis.shape[0])

    @property
    def synergy_dim(self) -> int:
        return int(self.basis.shape[1])

    @property
    def residual_dim(self) -> int:
        return int(self.residual_basis.shape[1])

    @property
    def policy_action_dim(self) -> int:
        return self.synergy_dim + self.residual_dim

    @property
    def residual_alpha(self) -> float:
        return float(self.body_synergy_contract.residual_alpha)

    @property
    def coefficient_bias(self) -> np.ndarray:
        ratio = np.clip(
            np.asarray(self.coefficient_center, dtype=np.float64)
            / np.asarray(self.coefficient_maximum, dtype=np.float64),
            1e-6,
            1.0 - 1e-6,
        )
        return (np.log(ratio) - np.log1p(-ratio)).astype(np.float32)

    @property
    def decoder_core_fingerprint(self) -> str:
        """Content identity of all values that affect numerical decoding."""

        return frozen_body_decoder_core_fingerprint(
            mode=self.body_synergy_contract.mode,
            actuator_names=self.actuator_names,
            residual_alpha=self.residual_alpha,
            **self._array_payload(),
        )

    @property
    def artifact_fingerprint(self) -> str:
        return canonical_json_sha256(self._identity_payload())

    def jax_params(self, *, dtype: Any = jnp.float32) -> FrozenBodyDecoderJaxParams:
        return FrozenBodyDecoderJaxParams(
            basis=jnp.asarray(self.basis, dtype=dtype),
            excitation_bounds=jnp.asarray(self.excitation_bounds, dtype=dtype),
            coefficient_maximum=jnp.asarray(self.coefficient_maximum, dtype=dtype),
            coefficient_bias=jnp.asarray(self.coefficient_bias, dtype=dtype),
            coefficient_temperature=jnp.asarray(self.coefficient_temperature, dtype=dtype),
            tonic_baseline=jnp.asarray(self.tonic_baseline, dtype=dtype),
            residual_basis=jnp.asarray(self.residual_basis, dtype=dtype),
            residual_alpha=jnp.asarray(self.residual_alpha, dtype=dtype),
        )

    def decode(self, raw_policy_action: Any) -> FrozenBodyDecoderOutput:
        raw = jnp.asarray(raw_policy_action)
        return decode_frozen_body_action(raw, self.jax_params(dtype=raw.dtype))

    def save(self, path: str | Path) -> Path:
        return save_frozen_body_decoder(path, self)

    def _array_payload(self) -> dict[str, np.ndarray]:
        return {
            "basis": self.basis,
            "excitation_bounds": self.excitation_bounds,
            "coefficient_maximum": self.coefficient_maximum,
            "coefficient_center": self.coefficient_center,
            "coefficient_temperature": self.coefficient_temperature,
            "tonic_baseline": self.tonic_baseline,
            "residual_basis": self.residual_basis,
        }

    def _identity_payload(self) -> dict[str, Any]:
        contract = self.body_synergy_contract
        return {
            "schema_version": FROZEN_BODY_DECODER_SCHEMA_VERSION,
            "decoder_core_fingerprint": self.decoder_core_fingerprint,
            "portable_decoder_core_fingerprint": (contract.portable_decoder_core_fingerprint),
        }


def save_frozen_body_decoder(
    path: str | Path,
    decoder: FrozenBodyDecoder,
) -> Path:
    """Write a self-contained frozen decoder directory."""

    if not isinstance(decoder, FrozenBodyDecoder):
        raise TypeError("decoder must be a FrozenBodyDecoder")
    _validate_contract_execution_binding(decoder)
    destination = _artifact_directory(path)
    destination.mkdir(parents=True, exist_ok=True)
    array_path = destination / FROZEN_BODY_DECODER_ARRAYS
    manifest_path = destination / FROZEN_BODY_DECODER_MANIFEST
    contract_path = destination / BODY_SYNERGY_CONTRACT_FILENAME
    np.savez_compressed(array_path, **decoder._array_payload())
    decoder.body_synergy_contract.save(contract_path)
    unsigned = {
        **decoder._identity_payload(),
        "artifact_fingerprint": decoder.artifact_fingerprint,
        # Exact runtime binding is audited and file-bound, but deliberately
        # excluded from the portable decoder identity.  A legal Stage-1 ->
        # Stage-2 model/runtime change must preserve this artifact fingerprint.
        "body_synergy_contract_fingerprint": (decoder.body_synergy_contract.contract_fingerprint),
        "array_file": FROZEN_BODY_DECODER_ARRAYS,
        "array_file_sha256": _file_sha256(array_path),
        "body_synergy_contract_file": BODY_SYNERGY_CONTRACT_FILENAME,
        "body_synergy_contract_file_sha256": _file_sha256(contract_path),
        "mode": decoder.body_synergy_contract.mode,
        "body_action_dim": decoder.body_action_dim,
        "policy_action_dim": decoder.policy_action_dim,
        "basis_rank": decoder.synergy_dim,
        "residual_dim": decoder.residual_dim,
        "residual_alpha": decoder.residual_alpha,
        "actuator_names": list(decoder.actuator_names),
        "actuator_schema_hash": actuator_schema_hash(decoder.actuator_names),
    }
    manifest = {
        **unsigned,
        "manifest_fingerprint": canonical_json_sha256(unsigned),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def load_frozen_body_decoder(
    path: str | Path,
    *,
    expected_actuator_names: Sequence[str] | None = None,
    expected_artifact_fingerprint: str | None = None,
    expected_portable_decoder_core_fingerprint: str | None = None,
) -> FrozenBodyDecoder:
    """Load and fully revalidate a serialized frozen decoder."""

    source = Path(path)
    manifest_path = (
        source
        if source.is_file() and source.name == FROZEN_BODY_DECODER_MANIFEST
        else _artifact_directory(source) / FROZEN_BODY_DECODER_MANIFEST
    )
    root = manifest_path.parent
    manifest = _strict_json_object(manifest_path)
    expected_fields = {
        "schema_version",
        "decoder_core_fingerprint",
        "body_synergy_contract_fingerprint",
        "portable_decoder_core_fingerprint",
        "array_file",
        "array_file_sha256",
        "body_synergy_contract_file",
        "body_synergy_contract_file_sha256",
        "mode",
        "body_action_dim",
        "policy_action_dim",
        "basis_rank",
        "residual_dim",
        "residual_alpha",
        "actuator_names",
        "actuator_schema_hash",
        "artifact_fingerprint",
        "manifest_fingerprint",
    }
    if set(manifest) != expected_fields:
        raise ValueError(
            "frozen body decoder manifest fields differ from schema: "
            f"missing={sorted(expected_fields - set(manifest))} "
            f"extra={sorted(set(manifest) - expected_fields)}"
        )
    if manifest["schema_version"] != FROZEN_BODY_DECODER_SCHEMA_VERSION:
        raise ValueError("unsupported frozen body decoder schema_version")
    supplied_manifest_fingerprint = _require_sha256(manifest.pop("manifest_fingerprint"), "manifest_fingerprint")
    if canonical_json_sha256(manifest) != supplied_manifest_fingerprint:
        raise ValueError("frozen body decoder manifest_fingerprint mismatch")
    supplied_artifact_fingerprint = _require_sha256(manifest["artifact_fingerprint"], "artifact_fingerprint")
    if expected_artifact_fingerprint not in (None, "") and (
        str(expected_artifact_fingerprint) != supplied_artifact_fingerprint
    ):
        raise ValueError("frozen body decoder expected artifact fingerprint mismatch")

    array_name = _safe_local_filename(manifest["array_file"], FROZEN_BODY_DECODER_ARRAYS)
    contract_name = _safe_local_filename(manifest["body_synergy_contract_file"], BODY_SYNERGY_CONTRACT_FILENAME)
    array_path = root / array_name
    contract_path = root / contract_name
    if _file_sha256(array_path) != _require_sha256(manifest["array_file_sha256"], "array_file_sha256"):
        raise ValueError("frozen body decoder array file fingerprint mismatch")
    if _file_sha256(contract_path) != _require_sha256(
        manifest["body_synergy_contract_file_sha256"],
        "body_synergy_contract_file_sha256",
    ):
        raise ValueError("frozen body decoder contract file fingerprint mismatch")

    contract = load_body_synergy_contract(contract_path)
    if contract.contract_fingerprint != manifest["body_synergy_contract_fingerprint"]:
        raise ValueError("frozen decoder BodySynergyContractV2 fingerprint mismatch")
    if contract.portable_decoder_core_fingerprint != manifest["portable_decoder_core_fingerprint"]:
        raise ValueError("frozen decoder portable contract fingerprint mismatch")
    if expected_portable_decoder_core_fingerprint not in (None, "") and (
        str(expected_portable_decoder_core_fingerprint) != contract.portable_decoder_core_fingerprint
    ):
        raise ValueError("frozen decoder expected portable contract fingerprint mismatch")

    with np.load(array_path, allow_pickle=False) as data:
        required_arrays = {
            "basis",
            "excitation_bounds",
            "coefficient_maximum",
            "coefficient_center",
            "coefficient_temperature",
            "tonic_baseline",
            "residual_basis",
        }
        if set(data.files) != required_arrays:
            raise ValueError(
                "frozen decoder arrays differ from schema: "
                f"missing={sorted(required_arrays - set(data.files))} "
                f"extra={sorted(set(data.files) - required_arrays)}"
            )
        arrays = {name: np.asarray(data[name]) for name in sorted(required_arrays)}
    decoder = FrozenBodyDecoder(body_synergy_contract=contract, **arrays)
    _validate_contract_execution_binding(decoder)
    if decoder.decoder_core_fingerprint != manifest["decoder_core_fingerprint"]:
        raise ValueError("frozen body decoder numerical core fingerprint mismatch")
    if decoder.artifact_fingerprint != supplied_artifact_fingerprint:
        raise ValueError("frozen body decoder artifact fingerprint mismatch")
    identity = decoder._identity_payload()
    if any(manifest[key] != value for key, value in identity.items()):
        raise ValueError("frozen body decoder identity differs from manifest")
    _validate_manifest_semantics(manifest, decoder)
    if (
        expected_actuator_names is not None
        and tuple(str(name) for name in expected_actuator_names) != decoder.actuator_names
    ):
        raise ValueError("frozen body decoder actuator names/order differ from runtime")
    return decoder


def _validate_manifest_semantics(manifest: Mapping[str, Any], decoder: FrozenBodyDecoder) -> None:
    expected = {
        "mode": decoder.body_synergy_contract.mode,
        "body_action_dim": decoder.body_action_dim,
        "policy_action_dim": decoder.policy_action_dim,
        "basis_rank": decoder.synergy_dim,
        "residual_dim": decoder.residual_dim,
        "residual_alpha": decoder.residual_alpha,
        "actuator_names": list(decoder.actuator_names),
        "actuator_schema_hash": actuator_schema_hash(decoder.actuator_names),
    }
    differences = [key for key, value in expected.items() if manifest[key] != value]
    if differences:
        raise ValueError(f"frozen body decoder semantic manifest mismatch: differing_fields={differences}")


def _validate_contract_execution_binding(decoder: FrozenBodyDecoder) -> None:
    contract = decoder.body_synergy_contract
    source_binding = contract.source_binding
    if not isinstance(source_binding, Mapping):
        raise ValueError("BodySynergyContractV2 source_binding must be an object")
    binding = source_binding.get(FROZEN_BODY_DECODER_EXECUTION_BINDING_FIELD)
    if not isinstance(binding, Mapping):
        raise ValueError("BodySynergyContractV2 is missing the frozen decoder execution binding")
    required = {
        "schema_version",
        "decoder_core_fingerprint",
        "contract_decoder_declarations",
        "binding_fingerprint",
    }
    if set(binding) != required:
        raise ValueError("frozen decoder execution binding fields differ from schema")
    if binding.get("schema_version") != FROZEN_BODY_DECODER_EXECUTION_BINDING_SCHEMA_VERSION:
        raise ValueError("unsupported frozen decoder execution binding schema")
    unsigned = {key: value for key, value in binding.items() if key != "binding_fingerprint"}
    if _require_sha256(binding.get("binding_fingerprint"), "binding_fingerprint") != canonical_json_sha256(unsigned):
        raise ValueError("frozen decoder execution binding fingerprint mismatch")
    if (
        _require_sha256(binding.get("decoder_core_fingerprint"), "decoder_core_fingerprint")
        != decoder.decoder_core_fingerprint
    ):
        raise ValueError(
            "BodySynergyContractV2 decoder execution fingerprint differs from the executable frozen decoder arrays"
        )

    declarations = binding.get("contract_decoder_declarations")
    expected_declarations = _contract_decoder_declarations(contract)
    if declarations != expected_declarations:
        raise ValueError("frozen decoder execution binding declarations differ from BodySynergyContractV2 fingerprints")


def _contract_decoder_declarations(
    contract: BodySynergyContractV2,
) -> dict[str, Any]:
    return {
        "mode": contract.mode,
        "body_action_dim": contract.body_action_dim,
        "policy_action_dim": contract.policy_action_dim,
        "actuator_schema_hash": contract.actuator_schema_hash,
        "control_range_hash": contract.control_range_hash,
        "basis_fingerprint": contract.basis_fingerprint,
        "runtime_basis_fingerprint": contract.runtime_basis_fingerprint,
        "basis_rank": contract.basis_rank,
        "coefficient_transform_fingerprint": (contract.coefficient_transform_fingerprint),
        "coefficient_statistics_fingerprint": (contract.coefficient_statistics_fingerprint),
        "tonic_baseline_fingerprint": contract.tonic_baseline_fingerprint,
        "residual_basis_fingerprint": contract.residual_basis_fingerprint,
        "residual_fit_contract_fingerprint": (contract.residual_fit_contract_fingerprint),
        "residual_allowed_muscle_mask_fingerprint": (contract.residual_allowed_muscle_mask_fingerprint),
        "residual_dim": contract.residual_dim,
        "residual_alpha": contract.residual_alpha,
    }


def bounded_synergy_coefficients(
    raw_coefficients: Any,
    *,
    maximum: Any,
    bias: Any,
    temperature: Any,
) -> Any:
    """Apply the canonical bounded coefficient transform."""

    raw = jnp.asarray(raw_coefficients)
    maximum_value = jnp.asarray(maximum, dtype=raw.dtype)
    bias_value = jnp.asarray(bias, dtype=raw.dtype)
    temperature_value = jnp.asarray(temperature, dtype=raw.dtype)
    return maximum_value * _stable_sigmoid(raw / temperature_value + bias_value)


def _stable_sigmoid(value: Any) -> Any:
    array = jnp.asarray(value)
    return jnp.asarray(0.5, dtype=array.dtype) * (jnp.tanh(array * 0.5) + 1.0)


def _finite_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"frozen decoder array must have finite shape {shape}, got {array.shape}")
    return array


def _artifact_directory(path: str | Path) -> Path:
    source = Path(path)
    if source.name == FROZEN_BODY_DECODER_MANIFEST:
        return source.parent
    return source


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 fingerprint")
    return text


def _optional_sha256(value: Any, name: str) -> str | None:
    return None if value is None else _require_sha256(value, name)


def _safe_local_filename(value: Any, expected: str) -> str:
    text = str(value)
    if text != expected or Path(text).name != text:
        raise ValueError(f"frozen decoder file binding must be exactly {expected!r}")
    return text


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise ValueError("frozen body decoder manifest must be a JSON object")
    return payload
