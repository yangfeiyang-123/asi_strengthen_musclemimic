"""Fixed-basis muscle-synergy decoder utilities.

The production latent runtime consumes normalized body actions in ``[-1, 1]``.
Synergy bases, however, are fitted in a non-negative excitation space.  This
module keeps that distinction explicit: the neural decoder first constructs a
bounded excitation, then maps it back to the normalized action ABI used by the
existing controller.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from musclemimic.distill.physical import (
    MUSCLE_EXCITATION_FORMULA,
    MUSCLE_EXCITATION_ROUNDOFF_POLICY,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
    validate_muscle_channel_contract,
    validate_unit_muscle_ctrlrange,
)
from musclemimic.latent_muscle.networks import _MLP
from musclemimic.synergy.frozen_decoder import (
    FrozenBodyDecoderJaxParams,
    decode_frozen_body_action,
    normalized_to_physical,
    physical_to_normalized,
)


class SynergyDecoderOutput(NamedTuple):
    """Common auxiliary output returned by direct and synergy decoders."""

    action: Any
    physical_excitation: Any
    synergy_coefficients: Any
    baseline_excitation: Any
    residual_excitation: Any


@dataclass(frozen=True)
class LoadedSynergyBasis:
    """Name-aligned, runtime-ready fixed synergy basis."""

    basis: np.ndarray
    actuator_names: tuple[str, ...]
    excitation_bounds: np.ndarray
    fingerprint: str
    manifest: dict[str, Any]
    source_path: str | None = None

    @property
    def action_dim(self) -> int:
        return int(self.basis.shape[0])

    @property
    def synergy_dim(self) -> int:
        return int(self.basis.shape[1])

    def to_checkpoint_payload(self) -> dict[str, Any]:
        return {
            "basis": np.asarray(self.basis, dtype=np.float32),
            "actuator_names": list(self.actuator_names),
            "excitation_bounds": np.asarray(self.excitation_bounds, dtype=np.float32),
            "fingerprint": str(self.fingerprint),
            "manifest": _jsonable(self.manifest),
            "source_path": self.source_path,
        }


class LatentSynergyDecoder(nn.Module):
    """Decode ``(state, z)`` through fixed non-negative muscle synergies.

    ``basis`` and ``excitation_bounds`` are call arguments rather than trainable
    module fields.  They are persisted as a separately fingerprinted checkpoint
    contract and therefore cannot accidentally drift through the optimizer.
    """

    action_dim: int
    synergy_dim: int
    hidden_layer_dims: Sequence[int] = (512, 256)
    activation: str = "tanh"
    use_layernorm: bool = False
    include_baseline: bool = False
    residual_indices: tuple[int, ...] = ()
    residual_alpha: float = 0.0
    baseline_init: float = 0.01

    @nn.compact
    def __call__(
        self,
        state,
        latent,
        basis,
        excitation_bounds,
        *,
        return_aux: bool = False,
    ):
        if int(self.action_dim) <= 0 or int(self.synergy_dim) <= 0:
            raise ValueError("synergy decoder dimensions must be positive")
        if float(self.residual_alpha) < 0.0:
            raise ValueError("residual_alpha must be non-negative")
        if len({int(index) for index in self.residual_indices}) != len(self.residual_indices):
            raise ValueError("residual_indices must be unique")
        if any(int(index) < 0 or int(index) >= int(self.action_dim) for index in self.residual_indices):
            raise ValueError("residual index is outside the decoder action dimension")

        basis = jnp.asarray(basis, dtype=jnp.float32)
        bounds = jnp.asarray(excitation_bounds, dtype=jnp.float32)
        if basis.shape != (int(self.action_dim), int(self.synergy_dim)):
            raise ValueError(
                "synergy basis shape mismatch: "
                f"expected={(int(self.action_dim), int(self.synergy_dim))}, got={basis.shape}"
            )
        if bounds.shape != (int(self.action_dim), 2):
            raise ValueError(
                f"excitation bounds shape mismatch: expected={(int(self.action_dim), 2)}, got={bounds.shape}"
            )

        x = jnp.concatenate([jnp.asarray(state), jnp.asarray(latent)], axis=-1)
        hidden = _MLP(
            hidden_layer_dims=self.hidden_layer_dims,
            activation=self.activation,
            use_layernorm=self.use_layernorm,
            name="trunk",
        )(x)
        coefficient_logits = nn.Dense(
            int(self.synergy_dim),
            kernel_init=orthogonal(0.01),
            bias_init=constant(_inverse_softplus(0.05)),
            name="synergy_coefficients",
        )(hidden)
        coefficients = nn.softplus(coefficient_logits)

        if self.include_baseline:
            # The tonic baseline is deliberately state-only.  Letting it reuse
            # the state+latent trunk would provide a full-action bypass around
            # the fixed synergy span and invalidate the scientific constraint.
            baseline_hidden = _MLP(
                hidden_layer_dims=self.hidden_layer_dims,
                activation=self.activation,
                use_layernorm=self.use_layernorm,
                name="baseline_trunk",
            )(jnp.asarray(state))
            baseline_logits = nn.Dense(
                int(self.action_dim),
                kernel_init=orthogonal(0.01),
                bias_init=constant(_inverse_softplus(float(self.baseline_init))),
                name="baseline",
            )(baseline_hidden)
            baseline = nn.softplus(baseline_logits)
        else:
            baseline = jnp.zeros((*hidden.shape[:-1], int(self.action_dim)), dtype=hidden.dtype)

        excitation = baseline + jnp.matmul(coefficients, basis.T)
        residual_full = jnp.zeros_like(excitation)
        if self.residual_indices:
            raw_residual = nn.Dense(
                len(self.residual_indices),
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
                name="residual",
            )(hidden)
            residual_values = float(self.residual_alpha) * jnp.tanh(raw_residual)
            residual_full = residual_full.at[..., jnp.asarray(self.residual_indices)].set(residual_values)
            excitation = excitation + residual_full

        lower = bounds[:, 0]
        upper = bounds[:, 1]
        excitation = jnp.clip(excitation, lower, upper)
        action = physical_to_normalized(excitation, bounds)
        output = SynergyDecoderOutput(
            action=action,
            physical_excitation=excitation,
            synergy_coefficients=coefficients,
            baseline_excitation=baseline,
            residual_excitation=residual_full,
        )
        return output if return_aux else output.action


class PortableLatentSynergyDecoder(nn.Module):
    """Predict only raw ``c/rho`` and execute the shared frozen decoder.

    Unlike :class:`LatentSynergyDecoder`, this production path has no learned
    354-D baseline and no direct-coordinate residual head.  The tonic vector,
    structured R basis, coefficient transform, alpha, and clipping bounds all
    come from one checkpointed :class:`FrozenBodyDecoder` artifact.
    """

    action_dim: int
    synergy_dim: int
    residual_dim: int = 0
    hidden_layer_dims: Sequence[int] = (512, 256)
    activation: str = "tanh"
    use_layernorm: bool = False

    @nn.compact
    def __call__(
        self,
        state,
        latent,
        decoder_params: FrozenBodyDecoderJaxParams,
        *,
        return_aux: bool = False,
    ):
        if int(self.action_dim) <= 0 or int(self.synergy_dim) <= 0:
            raise ValueError("portable synergy decoder dimensions must be positive")
        if int(self.residual_dim) < 0:
            raise ValueError("portable residual dimension must be non-negative")
        if tuple(decoder_params.basis.shape) != (
            int(self.action_dim),
            int(self.synergy_dim),
        ):
            raise ValueError("portable decoder W shape differs from network contract")
        if tuple(decoder_params.residual_basis.shape) != (
            int(self.action_dim),
            int(self.residual_dim),
        ):
            raise ValueError("portable decoder R shape differs from network contract")

        x = jnp.concatenate([jnp.asarray(state), jnp.asarray(latent)], axis=-1)
        hidden = _MLP(
            hidden_layer_dims=self.hidden_layer_dims,
            activation=self.activation,
            use_layernorm=self.use_layernorm,
            name="trunk",
        )(x)
        raw_coefficients = nn.Dense(
            int(self.synergy_dim),
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="raw_synergy_coefficients",
        )(hidden)
        if int(self.residual_dim):
            raw_residual = nn.Dense(
                int(self.residual_dim),
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
                name="raw_structured_residual",
            )(hidden)
            raw_action = jnp.concatenate([raw_coefficients, raw_residual], axis=-1)
        else:
            raw_action = raw_coefficients
        output = portable_decoder_output_from_raw(raw_action, decoder_params)
        return output if return_aux else output.action


def portable_decoder_output_from_raw(
    raw_action: Any,
    decoder_params: FrozenBodyDecoderJaxParams,
) -> SynergyDecoderOutput:
    """Latent-facing view of the shared raw c/rho decoder.

    This seam is intentionally network-free: tests, dataset QC, and runtime
    migration can prove that Stage-1 and latent coordinates decode identically
    before considering how a latent network predicts those coordinates.
    """

    decoded = decode_frozen_body_action(raw_action, decoder_params)
    raw = jnp.asarray(raw_action)
    tonic = jnp.asarray(decoder_params.tonic_baseline, dtype=raw.dtype)
    baseline = jnp.broadcast_to(tonic, (*raw.shape[:-1], int(tonic.shape[0])))
    return SynergyDecoderOutput(
        action=decoded.body_action,
        physical_excitation=decoded.physical_excitation,
        synergy_coefficients=decoded.synergy_coefficients,
        baseline_excitation=baseline,
        residual_excitation=decoded.residual_excitation,
    )


def load_fixed_synergy_basis(
    path: str | Path,
    *,
    expected_actuator_names: Sequence[str] | None = None,
    default_excitation_min: float = 0.0,
    default_excitation_max: float = 1.0,
    test_only_allow_legacy: bool = False,
) -> LoadedSynergyBasis:
    """Load a production artifact or an explicitly test-only legacy payload.

    Production always uses :func:`musclemimic.synergy.basis_artifact.load_synergy_basis`
    so fingerprint, split, teacher, dataset, transform, and ordered-name provenance
    are verified.  Raw NPZ/JSON is deliberately unavailable unless the caller
    sets the conspicuous test-only flag; such payloads are marked noncanonical.
    """

    source = Path(path)
    payload: Any = None
    legacy = source.is_file() and source.name != "manifest.json"
    if legacy and not bool(test_only_allow_legacy):
        raise ValueError(
            "production synergy decoder requires a formal basis artifact directory/manifest; raw NPZ/JSON is test-only"
        )
    if legacy:
        payload = _load_fallback_payload(source)
    else:
        try:
            from musclemimic.synergy.basis_artifact import load_synergy_basis

            payload = load_synergy_basis(source)
        except (ImportError, ModuleNotFoundError):
            payload = _load_fallback_payload(source)

    loaded = coerce_synergy_basis(
        payload,
        expected_actuator_names=expected_actuator_names,
        default_excitation_min=default_excitation_min,
        default_excitation_max=default_excitation_max,
        source_path=str(source.resolve()),
    )
    if legacy:
        manifest = dict(loaded.manifest)
        manifest.update(
            {
                "artifact_kind": "legacy_noncanonical",
                "noncanonical": True,
                "test_only_allow_legacy": True,
            }
        )
        loaded = coerce_synergy_basis(
            loaded.to_checkpoint_payload() | {"manifest": manifest},
            expected_actuator_names=expected_actuator_names,
            source_path=str(source.resolve()),
        )
    return loaded


def coerce_synergy_basis(
    payload: Any,
    *,
    expected_actuator_names: Sequence[str] | None = None,
    default_excitation_min: float = 0.0,
    default_excitation_max: float = 1.0,
    source_path: str | None = None,
) -> LoadedSynergyBasis:
    """Normalize dict/object/checkpoint payloads into one strict contract."""

    mapping = _as_mapping(payload)
    basis_value = mapping.get("basis", mapping.get("W"))
    if basis_value is None:
        raise ValueError("synergy artifact is missing basis/W")
    basis = np.asarray(basis_value, dtype=np.float32)
    if basis.ndim != 2 or basis.shape[0] <= 0 or basis.shape[1] <= 0:
        raise ValueError(f"synergy basis must be a non-empty [muscle, rank] matrix, got {basis.shape}")
    if not np.all(np.isfinite(basis)) or np.any(basis < -1e-7):
        raise ValueError("synergy basis must be finite and non-negative")
    basis = np.maximum(basis, 0.0)

    names_value = mapping.get("actuator_names", mapping.get("muscle_names"))
    if names_value is None:
        raise ValueError("synergy artifact is missing ordered actuator/muscle names")
    names = tuple(str(name) for name in np.asarray(names_value).tolist())
    if len(names) != basis.shape[0] or len(set(names)) != len(names):
        raise ValueError("synergy artifact names must be unique and match basis rows")

    bounds_value = mapping.get("excitation_bounds")
    if bounds_value is None:
        lower = float(mapping.get("excitation_min", default_excitation_min))
        upper = float(mapping.get("excitation_max", default_excitation_max))
        bounds = np.tile(np.asarray([[lower, upper]], dtype=np.float32), (basis.shape[0], 1))
    else:
        bounds = np.asarray(bounds_value, dtype=np.float32)
        if bounds.shape == (2,):
            bounds = np.tile(bounds[None, :], (basis.shape[0], 1))
    if bounds.shape != (basis.shape[0], 2):
        raise ValueError("synergy excitation_bounds must have shape [muscle, 2]")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("synergy excitation bounds must be finite and increasing")
    if np.any(bounds[:, 0] < -1e-7):
        raise ValueError(
            "synergy excitation bounds must be non-negative; signed data.ctrl requires an explicit excitation conversion"
        )

    expected = None if expected_actuator_names is None else tuple(str(name) for name in expected_actuator_names)
    if expected is not None:
        if names != expected:
            missing = sorted(set(expected) - set(names))
            extra = sorted(set(names) - set(expected))
            raise ValueError(
                "synergy basis actuator names/order differ from decoder body schema: "
                f"missing={missing}, extra={extra}; reordering is forbidden"
            )

    manifest_value = mapping.get("manifest")
    manifest = dict(manifest_value) if isinstance(manifest_value, Mapping) else {}
    manifest.update(
        {
            "schema_version": str(manifest.get("schema_version", "latent_fixed_synergy_basis_v2")),
            "actuator_names": list(names),
            "rank": int(basis.shape[1]),
            "action_dim": int(basis.shape[0]),
            "signal_kind": str(manifest.get("signal_kind", mapping.get("signal_kind", "physical_excitation"))),
        }
    )
    fingerprint = _basis_fingerprint(basis, names, bounds, manifest)
    supplied = mapping.get("fingerprint")
    if supplied is not None:
        manifest.setdefault("source_fingerprint", str(supplied))
    manifest["runtime_fingerprint"] = fingerprint
    return LoadedSynergyBasis(
        basis=basis,
        actuator_names=names,
        excitation_bounds=bounds,
        fingerprint=fingerprint,
        manifest=manifest,
        source_path=source_path or mapping.get("source_path"),
    )


def _load_fallback_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"synergy basis artifact not found: {path}")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    if path.suffix.lower() == ".json":
        mapping = _strict_json_load(path)
        basis_path = mapping.get("basis_path")
        if basis_path is not None and "basis" not in mapping and "W" not in mapping:
            resolved = Path(basis_path)
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            nested = _load_fallback_payload(resolved)
            nested.update(mapping)
            return nested
        return mapping
    raise ValueError(f"unsupported synergy basis artifact: {path}")


def validate_decoder_synergy_basis(
    basis: LoadedSynergyBasis,
    *,
    allow_noncanonical: bool = False,
) -> None:
    """Enforce excitation and full regional-composite semantics for decoding."""

    manifest = dict(basis.manifest)
    if manifest.get("noncanonical") is True:
        if not allow_noncanonical:
            raise ValueError("noncanonical synergy basis is forbidden outside explicit tests")
        return
    try:
        from musclemimic.synergy import schema as synergy_schema

        excitation_signal_kind = synergy_schema.EXCITATION_SIGNAL_KIND
    except (ImportError, ModuleNotFoundError):
        excitation_signal_kind = "physical_excitation_unit"
    if manifest.get("signal_kind") != excitation_signal_kind:
        raise ValueError(
            "latent synergy decoder requires a physical-excitation basis; "
            f"got signal_kind={manifest.get('signal_kind')!r}"
        )
    transform = manifest.get("transform")
    if not isinstance(transform, Mapping):
        raise ValueError("latent synergy decoder requires explicit v2 excitation transform provenance")
    expected_transform = {
        "kind": UNIT_EXCITATION_TRANSFORM,
        "formula": MUSCLE_EXCITATION_FORMULA,
        "roundoff_policy": MUSCLE_EXCITATION_ROUNDOFF_POLICY,
        "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
    }
    for field, expected in expected_transform.items():
        if transform.get(field) != expected:
            raise ValueError(f"latent synergy decoder requires v2 excitation transform {field}={expected!r}")
    transform_names = tuple(str(name) for name in transform.get("actuator_names", ()))
    if transform_names != basis.actuator_names:
        raise ValueError("latent synergy decoder excitation transform actuator order mismatch")
    validate_unit_muscle_ctrlrange(
        basis.actuator_names,
        transform.get("ctrlrange"),
    )
    validate_muscle_channel_contract(
        transform.get("muscle_channel_contract"),
        expected_names=basis.actuator_names,
    )
    expected_bounds = np.broadcast_to(
        np.asarray([0.0, 1.0], dtype=np.float32),
        basis.excitation_bounds.shape,
    )
    if not np.array_equal(basis.excitation_bounds, expected_bounds):
        raise ValueError("latent synergy decoder v2 requires exact [0,1] excitation bounds")
    if str(manifest.get("region")) == "regional_composite":
        _validate_regional_composite(basis)


def _validate_regional_composite(basis: LoadedSynergyBasis) -> None:
    manifest = basis.manifest
    if manifest.get("composite_schema_version") != "regional_synergy_composite_v1":
        raise ValueError("regional composite synergy basis has an unsupported schema")
    regions = manifest.get("composite_regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("regional composite synergy basis requires composite_regions")
    if not str(manifest.get("regional_grouping_fingerprint", "")).strip():
        raise ValueError("regional composite is missing regional_grouping_fingerprint")
    row_owner: set[int] = set()
    column_cursor = 0
    matrix = np.asarray(basis.basis)
    for item in regions:
        if not isinstance(item, Mapping):
            raise ValueError("regional composite entries must be objects")
        rows = [int(value) for value in item.get("row_indices", ())]
        names = [str(value) for value in item.get("muscle_names", ())]
        start = int(item.get("column_start", -1))
        stop = int(item.get("column_stop", -1))
        rank = int(item.get("rank", -1))
        if (
            not rows
            or len(rows) != len(set(rows))
            or any(row < 0 or row >= basis.action_dim for row in rows)
            or row_owner.intersection(rows)
        ):
            raise ValueError("regional composite row ownership is invalid or overlapping")
        if names != [basis.actuator_names[row] for row in rows]:
            raise ValueError("regional composite muscle names/order differ from row_indices")
        if start != column_cursor or stop - start != rank or rank <= 0:
            raise ValueError("regional composite column slices must be contiguous and rank-aligned")
        if not str(item.get("component_artifact_fingerprint", "")).strip():
            raise ValueError("regional composite component is missing artifact fingerprint")
        outside = np.ones(basis.action_dim, dtype=bool)
        outside[np.asarray(rows, dtype=np.int64)] = False
        if np.any(np.abs(matrix[outside, start:stop]) > 1e-8):
            raise ValueError("regional composite basis is not block diagonal")
        row_owner.update(rows)
        column_cursor = stop
    if row_owner != set(range(basis.action_dim)) or column_cursor != basis.synergy_dim:
        raise ValueError("regional composite does not exactly cover all rows and columns")


def _strict_json_load(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise ValueError(f"legacy synergy JSON must contain an object: {path}")
    return payload


def _as_mapping(payload: Any) -> dict[str, Any]:
    if isinstance(payload, LoadedSynergyBasis):
        return payload.to_checkpoint_payload()
    if isinstance(payload, Mapping):
        return dict(payload)
    result: dict[str, Any] = {}
    for key in (
        "basis",
        "W",
        "actuator_names",
        "muscle_names",
        "excitation_bounds",
        "excitation_min",
        "excitation_max",
        "fingerprint",
        "manifest",
        "signal_kind",
        "source_path",
    ):
        if hasattr(payload, key):
            result[key] = getattr(payload, key)
    return result


def _basis_fingerprint(
    basis: np.ndarray,
    actuator_names: Sequence[str],
    excitation_bounds: np.ndarray,
    manifest: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("basis", np.asarray(basis, dtype="<f4")),
        ("bounds", np.asarray(excitation_bounds, dtype="<f4")),
    ):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    semantic_manifest = {
        key: value
        for key, value in dict(manifest).items()
        if key not in {"runtime_fingerprint", "source_fingerprint", "source_path", "path"}
    }
    digest.update(
        json.dumps(
            {"actuator_names": list(actuator_names), "manifest": _jsonable(semantic_manifest)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _inverse_softplus(value: float) -> float:
    if float(value) <= 0.0:
        raise ValueError("softplus initialization must be positive")
    return float(np.log(np.expm1(float(value))))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
