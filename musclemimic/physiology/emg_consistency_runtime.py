"""Fail-closed Stage-1 PEASD-Lite runtime binding.

The phase-reference tube and its observation mapping form one inseparable
training asset.  This module validates that bundle before environment creation,
then compiles it against the concrete MuJoCo muscle layout once the environment
exists.  Reward code receives only the resulting static JAX arrays and scalar
configuration; it never performs file I/O inside a transformed function.

The only temporal coordinate in this contract is normalized trajectory
progress.  In particular, no event, impact, or named-phase semantics are
inferred here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.physiology.emg_anchor import (
    DEFAULT_HUBER_DELTA,
    DEFAULT_TUBE_KAPPA,
    EmgAnchorSpec,
    EmgAnchorSpecIdentity,
    build_emg_anchor_spec,
    load_json_mapping,
)
from musclemimic.physiology.emg_reference import (
    EmgPhaseReferenceTube,
    load_emg_phase_reference_tube,
    resolve_emg_reference_reward_gate,
)
from musclemimic.physiology.runtime_binding import resolve_ordered_policy_muscle_layout

EMG_OBSERVATION_MAPPING_FILENAME = "emg_observation_mapping.json"
EMG_CONSISTENCY_PREFLIGHT_SCHEMA_VERSION = "stage1_peasd_lite_preflight_contract_v1"
EMG_CONSISTENCY_RUNTIME_SCHEMA_VERSION = "stage1_peasd_lite_runtime_contract_v1"
EMG_CONSISTENCY_ARMS = ("T1", "T2", "T3", "T4")
EMG_CONSISTENCY_DISABLED_ARM = "T0"
EMG_CONSISTENCY_SIGNAL = "mujoco_scalar_activation_state"
EMG_CONSISTENCY_PHASE_COORDINATE = "normalized_trajectory_progress"

_ALLOWED_CONFIG_KEYS = frozenset(
    {
        "enabled",
        "mode",
        "arm",
        "action_id",
        "reference_cache",
        "mapping_path",
        "expected_reference_fingerprint",
        "expected_mapping_sha256",
        "anchor_weight_max",
        "synergy_weight_max",
        "start_update",
        "ramp_updates",
        "tube_kappa",
        "huber_delta",
        "anchor_max_penalty_each",
        "synergy_max_penalty_each",
        "synergy_shape_weight",
        "synergy_intensity_weight",
        "synergy_phase_shuffle_offset_bins",
        "use_activation",
    }
)


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"emg_consistency.{field} must be a lowercase SHA-256 digest")
    return text


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {str(key): item for key, item in value.items()}
    raise ValueError("reward_params.emg_consistency must be a mapping")


def _finite_nonnegative(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"emg_consistency.{field} must be finite and non-negative")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"emg_consistency.{field} must be finite and non-negative")
    return number


def _finite_positive(value: Any, *, field: str) -> float:
    number = _finite_nonnegative(value, field=field)
    if number <= 0.0:
        raise ValueError(f"emg_consistency.{field} must be positive")
    return number


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"emg_consistency.{field} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"emg_consistency.{field} must be a non-negative integer") from error
    if number < 0 or float(value) != float(number):
        raise ValueError(f"emg_consistency.{field} must be a non-negative integer")
    return number


def _resolve_path(value: Any, *, field: str, base_dir: str | Path) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"emg_consistency.{field} is required when enabled=true")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(base_dir).expanduser().resolve() / path
    return path.resolve(strict=True)


@dataclass(frozen=True)
class EmgConsistencyConfig:
    """Validated scalar configuration for one Stage-1 treatment arm."""

    arm: str
    mode: str
    action_id: str
    reference_cache: Path
    mapping_path: Path | None
    expected_reference_fingerprint: str | None
    expected_mapping_sha256: str | None
    anchor_weight_max: float
    synergy_weight_max: float
    start_update: int
    ramp_updates: int
    tube_kappa: float
    huber_delta: float
    anchor_max_penalty_each: float
    synergy_max_penalty_each: float
    synergy_shape_weight: float
    synergy_intensity_weight: float
    synergy_phase_shuffle_offset_bins: int

    @property
    def synergy_phase_shuffled(self) -> bool:
        return self.arm == "T4"

    @property
    def training_signal_enabled(self) -> bool:
        return self.mode == "reward"


@dataclass(frozen=True)
class EmgReferenceBundle:
    """One verified tube plus the exact mapping bytes shipped beside it."""

    root: Path
    tube: EmgPhaseReferenceTube
    mapping: dict[str, Any]
    mapping_path: Path
    mapping_sha256: str


@dataclass(frozen=True)
class EmgConsistencyRuntime:
    """Static arrays and content identity consumed by ``MimicReward``."""

    config: EmgConsistencyConfig
    bundle: EmgReferenceBundle
    spec: EmgAnchorSpec
    spec_identity: EmgAnchorSpecIdentity
    action_index: int
    contract: dict[str, Any]


def validate_emg_consistency_config(
    raw_config: Any,
    *,
    base_dir: str | Path,
) -> EmgConsistencyConfig | None:
    """Validate the new ``reward_params.emg_consistency`` schema.

    ``None`` and an explicit ``T0`` are reward-neutral.  Enabled arms are
    intentionally exact: T1 activation-only, T2 synergy-only, T3 both, and T4
    real-phase activation plus deterministically shifted synergy bins.
    """

    config = _plain_mapping(raw_config)
    unknown = sorted(set(config) - _ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"reward_params.emg_consistency has unsupported keys: {unknown}")
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("emg_consistency.enabled must be boolean")

    arm = str(config.get("arm", EMG_CONSISTENCY_DISABLED_ARM)).strip().upper()
    mode = str(config.get("mode", "reward" if enabled else "off")).strip().lower()
    anchor_weight = _finite_nonnegative(config.get("anchor_weight_max", 0.0), field="anchor_weight_max")
    synergy_weight = _finite_nonnegative(config.get("synergy_weight_max", 0.0), field="synergy_weight_max")
    shuffle_offset = _nonnegative_int(
        config.get("synergy_phase_shuffle_offset_bins", 0),
        field="synergy_phase_shuffle_offset_bins",
    )
    use_activation = config.get("use_activation", True)
    if not isinstance(use_activation, bool) or use_activation is not True:
        raise ValueError("emg_consistency.use_activation must remain true; policy action is not activation")

    if not enabled:
        if mode != "off":
            raise ValueError("disabled emg_consistency must declare mode=off")
        if arm != EMG_CONSISTENCY_DISABLED_ARM:
            raise ValueError("disabled emg_consistency must declare arm=T0")
        if anchor_weight != 0.0 or synergy_weight != 0.0 or shuffle_offset != 0:
            raise ValueError("disabled T0 emg_consistency must have zero weights and no shuffle")
        if str(config.get("reference_cache", "") or "").strip() or str(config.get("mapping_path", "") or "").strip():
            raise ValueError("disabled T0 emg_consistency must not bind an EMG reference")
        return None

    if mode not in {"reward", "diagnostics_only"}:
        raise ValueError("enabled emg_consistency.mode must be reward or diagnostics_only")
    if mode == "reward" and arm not in EMG_CONSISTENCY_ARMS:
        raise ValueError(f"reward emg_consistency.arm must be one of {list(EMG_CONSISTENCY_ARMS)}")
    if mode == "diagnostics_only" and arm != EMG_CONSISTENCY_DISABLED_ARM:
        raise ValueError("diagnostics_only emg_consistency is reserved for post-hoc T0 evaluation")
    action_id = str(config.get("action_id", "") or "").strip()
    if not action_id:
        raise ValueError("emg_consistency.action_id is required when enabled=true")
    expected_weights = (
        (False, False)
        if mode == "diagnostics_only"
        else {
            "T1": (True, False),
            "T2": (False, True),
            "T3": (True, True),
            "T4": (True, True),
        }[arm]
    )
    active_weights = (anchor_weight > 0.0, synergy_weight > 0.0)
    if active_weights != expected_weights:
        raise ValueError(
            f"emg_consistency {arm} requires activation/synergy weight activity "
            f"{expected_weights}, got {active_weights}"
        )
    if arm == "T4" and shuffle_offset <= 0:
        raise ValueError("emg_consistency T4 requires a positive deterministic synergy phase-bin offset")
    if arm != "T4" and shuffle_offset != 0:
        raise ValueError(f"emg_consistency {arm} forbids synergy phase shuffling")

    reference_cache = _resolve_path(
        config.get("reference_cache"),
        field="reference_cache",
        base_dir=base_dir,
    )
    mapping_text = str(config.get("mapping_path", "") or "").strip()
    mapping_path = _resolve_path(mapping_text, field="mapping_path", base_dir=base_dir) if mapping_text else None
    expected_reference = str(config.get("expected_reference_fingerprint", "") or "").strip() or None
    expected_mapping = str(config.get("expected_mapping_sha256", "") or "").strip() or None
    if expected_reference is not None:
        expected_reference = _require_sha256(expected_reference, field="expected_reference_fingerprint")
    if expected_mapping is not None:
        expected_mapping = _require_sha256(expected_mapping, field="expected_mapping_sha256")

    shape_weight = _finite_nonnegative(
        config.get("synergy_shape_weight", 1.0),
        field="synergy_shape_weight",
    )
    intensity_weight = _finite_nonnegative(
        config.get("synergy_intensity_weight", 0.25),
        field="synergy_intensity_weight",
    )
    if shape_weight <= 0.0 and intensity_weight <= 0.0:
        raise ValueError("emg_consistency synergy diagnostics require a positive shape or intensity weight")
    if intensity_weight > shape_weight:
        raise ValueError("emg_consistency synergy_intensity_weight must not exceed synergy_shape_weight")

    return EmgConsistencyConfig(
        arm=arm,
        mode=mode,
        action_id=action_id,
        reference_cache=reference_cache,
        mapping_path=mapping_path,
        expected_reference_fingerprint=expected_reference,
        expected_mapping_sha256=expected_mapping,
        anchor_weight_max=anchor_weight,
        synergy_weight_max=synergy_weight,
        start_update=_nonnegative_int(config.get("start_update", 0), field="start_update"),
        ramp_updates=_nonnegative_int(config.get("ramp_updates", 0), field="ramp_updates"),
        tube_kappa=_finite_positive(config.get("tube_kappa", DEFAULT_TUBE_KAPPA), field="tube_kappa"),
        huber_delta=_finite_positive(config.get("huber_delta", DEFAULT_HUBER_DELTA), field="huber_delta"),
        anchor_max_penalty_each=_finite_positive(
            config.get("anchor_max_penalty_each", 1.0),
            field="anchor_max_penalty_each",
        ),
        synergy_max_penalty_each=_finite_positive(
            config.get("synergy_max_penalty_each", 1.0),
            field="synergy_max_penalty_each",
        ),
        synergy_shape_weight=shape_weight,
        synergy_intensity_weight=intensity_weight,
        synergy_phase_shuffle_offset_bins=shuffle_offset,
    )


def load_verified_emg_reference_bundle(config: EmgConsistencyConfig) -> EmgReferenceBundle:
    """Load a training-cleared tube and its content-bound local mapping."""

    tube = load_emg_phase_reference_tube(config.reference_cache)
    resolve_emg_reference_reward_gate(tube, enabled=True)
    root = config.reference_cache.parent if config.reference_cache.is_file() else config.reference_cache
    root = root.resolve(strict=True)
    mapping_path = (root / EMG_OBSERVATION_MAPPING_FILENAME).resolve(strict=True)
    if config.mapping_path is not None and config.mapping_path != mapping_path:
        raise ValueError(
            "emg_consistency.mapping_path must name the mapping bundled beside the tube; "
            f"configured={config.mapping_path} bundled={mapping_path}"
        )
    mapping_sha256 = _sha256_file(mapping_path)
    declared_sha256 = _require_sha256(
        tube.mapping_binding.get("mapping_sha256"),
        field="tube.mapping_binding.mapping_sha256",
    )
    if mapping_sha256 != declared_sha256:
        raise ValueError(
            "EMG reference bundled mapping SHA-256 differs from the tube binding: "
            f"tube={declared_sha256} bundled={mapping_sha256}"
        )
    if config.expected_mapping_sha256 is not None and mapping_sha256 != config.expected_mapping_sha256:
        raise ValueError("emg_consistency.expected_mapping_sha256 differs from the bundled mapping")
    if (
        config.expected_reference_fingerprint is not None
        and tube.reference_fingerprint != config.expected_reference_fingerprint
    ):
        raise ValueError("emg_consistency.expected_reference_fingerprint differs from the loaded tube")

    mapping = load_json_mapping(mapping_path)
    mapping_id = str(mapping.get("mapping_id", "") or "").strip()
    if mapping_id != str(tube.mapping_binding.get("mapping_id", "") or "").strip():
        raise ValueError("EMG reference bundled mapping_id differs from the tube binding")
    if str(mapping.get("review_status", "") or "").strip().lower() != "verified":
        raise ValueError("EMG reference bundled mapping review_status must be verified")
    if mapping.get("training_enabled") is not True:
        raise ValueError("EMG reference bundled mapping must set training_enabled=true")
    evidence = mapping.get("review_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("EMG reference bundled mapping must retain non-empty review_evidence")

    comparable = 0
    for index, channel in enumerate(mapping.get("channels", ())):
        if not isinstance(channel, Mapping):
            raise ValueError(f"EMG bundled mapping channel[{index}] must be a mapping")
        status = str(channel.get("mapping_status", "") or "").strip().lower()
        if status == "excluded_no_verified_model_homolog":
            if not str(channel.get("exclusion_reason", "") or "").strip():
                raise ValueError("excluded EMG mapping channels require a traceable exclusion_reason")
            continue
        if status != "mapped":
            raise ValueError(f"EMG bundled mapping channel[{index}] has unsupported status {status!r}")
        confidence = str(channel.get("mapping_confidence", "") or "").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            raise ValueError(f"EMG bundled mapping channel[{index}] retains provisional confidence")
        comparable += 1
    if comparable != tube.channel_count:
        raise ValueError(
            f"EMG bundled mapping exposes {comparable} comparable channels but the tube has {tube.channel_count}"
        )
    if config.action_id not in tube.action_ids:
        raise ValueError(
            f"emg_consistency.action_id={config.action_id!r} is absent from tube actions {list(tube.action_ids)}"
        )
    if config.synergy_phase_shuffle_offset_bins >= tube.phase_bin_count:
        raise ValueError(
            "emg_consistency.synergy_phase_shuffle_offset_bins must be smaller than the tube phase-bin count"
        )
    if config.synergy_phase_shuffled:
        half_cycle_offset = tube.phase_bin_count // 2
        if tube.phase_bin_count < 2 or config.synergy_phase_shuffle_offset_bins != half_cycle_offset:
            raise ValueError(
                "emg_consistency T4 must use a deterministic half-cycle circular phase shift: "
                f"expected offset {half_cycle_offset} for {tube.phase_bin_count} bins"
            )
    if tube.array_bundle_sha256 is None:
        raise ValueError("training-enabled EMG reference must bind its array bundle SHA-256")

    return EmgReferenceBundle(
        root=root,
        tube=tube,
        mapping=mapping,
        mapping_path=mapping_path,
        mapping_sha256=mapping_sha256,
    )


def build_emg_consistency_preflight_contract(
    raw_config: Any,
    *,
    base_dir: str | Path,
) -> dict[str, Any] | None:
    """Bind reviewed artifact identity before W&B/environment/GPU work."""

    config = validate_emg_consistency_config(raw_config, base_dir=base_dir)
    if config is None:
        return None
    bundle = load_verified_emg_reference_bundle(config)
    tube = bundle.tube
    trial_qc_review = tube.provenance["trial_qc_review"]
    normalization_actions = tube.normalization_binding["actions"]
    action_normalization = normalization_actions[tube.action_index(config.action_id)]
    mvc_quality_counts: dict[str, int] = {}
    for stats in action_normalization["channels"]:
        quality = str(stats["mvc_quality"])
        mvc_quality_counts[quality] = mvc_quality_counts.get(quality, 0) + 1
    payload = {
        "schema_version": EMG_CONSISTENCY_PREFLIGHT_SCHEMA_VERSION,
        "enabled": True,
        "arm": config.arm,
        "mode": config.mode,
        "training_signal_enabled": config.training_signal_enabled,
        "action_id": config.action_id,
        "action_index": tube.action_index(config.action_id),
        "reference_path": str(bundle.root),
        "reference_id": tube.reference_id,
        "reference_fingerprint": tube.reference_fingerprint,
        "array_bundle_sha256": _require_sha256(
            tube.array_bundle_sha256,
            field="tube.array_bundle_sha256",
        ),
        "reference_review_status": tube.review_status,
        "reference_training_enabled": bool(tube.training_enabled),
        "trial_qc_review_schema_version": str(trial_qc_review["schema_version"]),
        "trial_qc_review_sha256": _require_sha256(
            trial_qc_review["review_sha256"],
            field="tube.provenance.trial_qc_review.review_sha256",
        ),
        "mapping_path": str(bundle.mapping_path),
        "mapping_id": str(bundle.mapping.get("mapping_id", "")),
        "mapping_sha256": bundle.mapping_sha256,
        "mapping_review_status": str(bundle.mapping.get("review_status", "")),
        "phase_coordinate": EMG_CONSISTENCY_PHASE_COORDINATE,
        "signal": EMG_CONSISTENCY_SIGNAL,
        "phase_bin_count": int(tube.phase_bin_count),
        "channel_count": int(tube.channel_count),
        "synergy_count": int(tube.synergy_count),
        "normalization_schema_version": str(tube.normalization_binding["schema_version"]),
        "audit_normalization": str(tube.normalization_binding["audit_normalization"]),
        "model_normalization": str(tube.normalization_binding["model_normalization"]),
        "normalization_training_cohort_sha256": str(action_normalization["training_cohort_sha256"]),
        "mvc_quality_counts": mvc_quality_counts,
        "minimum_amplitude_confidence": float(np.min(tube.amplitude_confidence[tube.action_index(config.action_id)])),
        "maximum_task_p99_over_mvc": float(
            max(float(stats["task_p99_over_mvc"]) for stats in action_normalization["channels"])
        ),
        "anchor_weight_max": config.anchor_weight_max,
        "synergy_weight_max": config.synergy_weight_max,
        "anchor_max_penalty_each": config.anchor_max_penalty_each,
        "synergy_max_penalty_each": config.synergy_max_penalty_each,
        "start_update": config.start_update,
        "ramp_updates": config.ramp_updates,
        "tube_kappa": config.tube_kappa,
        "huber_delta": config.huber_delta,
        "synergy_shape_weight": config.synergy_shape_weight,
        "synergy_intensity_weight": config.synergy_intensity_weight,
        "synergy_phase_shuffled": config.synergy_phase_shuffled,
        "synergy_phase_shift_strategy": ("half_cycle_circular" if config.synergy_phase_shuffled else "none"),
        "synergy_phase_shuffle_offset_bins": config.synergy_phase_shuffle_offset_bins,
    }
    payload["binding_sha256"] = _canonical_json_sha256(payload)
    return payload


def compile_emg_consistency_runtime(
    env: Any,
    raw_config: Any,
    *,
    base_dir: str | Path,
) -> EmgConsistencyRuntime | None:
    """Compile the reviewed bundle against the concrete ordered muscle ABI."""

    config = validate_emg_consistency_config(raw_config, base_dir=base_dir)
    if config is None:
        return None
    bundle = load_verified_emg_reference_bundle(config)
    layout = resolve_ordered_policy_muscle_layout(env, model=env._model)
    if layout.width != 354:
        raise ValueError(f"Stage1 PEASD-Lite requires the full 354-muscle ABI, found {layout.width}")

    tube_schema_hash = str(bundle.tube.mapping_binding.get("actuator_schema_hash", "") or "").strip()
    if tube_schema_hash != layout.actuator_schema_hash:
        raise ValueError("EMG reference actuator schema hash differs from the runtime muscle-name order")
    model_binding = bundle.mapping.get("model_binding")
    if not isinstance(model_binding, Mapping):
        raise ValueError("EMG bundled mapping lacks model_binding")
    mapping_schema_hash = str(model_binding.get("actuator_schema_hash", "") or "").strip()
    if mapping_schema_hash != layout.actuator_schema_hash:
        raise ValueError("EMG bundled mapping actuator schema hash differs from the runtime muscle-name order")
    mapping_runtime_hash = str(model_binding.get("runtime_model_hash", "") or "").strip()
    if mapping_runtime_hash != layout.runtime_model_hash:
        raise ValueError("EMG bundled mapping runtime_model_hash differs from the concrete MuJoCo model")

    spec, identity = build_emg_anchor_spec(
        bundle.tube,
        bundle.mapping,
        actuator_names=layout.actuator_names,
        activation_addresses=layout.activation_addresses,
        muscle_channel_core_fingerprint=layout.muscle_channel_core_fingerprint,
        tube_kappa=config.tube_kappa,
        huber_delta=config.huber_delta,
        synergy_shape_weight=config.synergy_shape_weight,
        synergy_intensity_weight=config.synergy_intensity_weight,
    )
    action_index = bundle.tube.action_index(config.action_id)
    matched_core = {
        "mode": config.mode,
        "training_signal_enabled": config.training_signal_enabled,
        "reference_fingerprint": bundle.tube.reference_fingerprint,
        "array_bundle_sha256": bundle.tube.array_bundle_sha256,
        "mapping_sha256": bundle.mapping_sha256,
        "action_id": config.action_id,
        "action_index": action_index,
        "runtime_model_hash": layout.runtime_model_hash,
        "actuator_schema_hash": layout.actuator_schema_hash,
        "muscle_channel_core_fingerprint": layout.muscle_channel_core_fingerprint,
        "anchor_loss_spec_fingerprint": identity.loss_spec_fingerprint,
        "anchor_weight_max": config.anchor_weight_max,
        "synergy_weight_max": config.synergy_weight_max,
        "anchor_max_penalty_each": config.anchor_max_penalty_each,
        "synergy_max_penalty_each": config.synergy_max_penalty_each,
        "start_update": config.start_update,
        "ramp_updates": config.ramp_updates,
    }
    preflight = build_emg_consistency_preflight_contract(raw_config, base_dir=base_dir)
    assert preflight is not None
    contract = {
        **{key: value for key, value in preflight.items() if key != "binding_sha256"},
        "schema_version": EMG_CONSISTENCY_RUNTIME_SCHEMA_VERSION,
        "ordered_actuator_count": layout.width,
        "actuator_schema_hash": layout.actuator_schema_hash,
        "runtime_model_hash": layout.runtime_model_hash,
        "muscle_channel_core_fingerprint": layout.muscle_channel_core_fingerprint,
        "anchor_loss_spec_fingerprint": identity.loss_spec_fingerprint,
        "matched_reward_core_fingerprint": _canonical_json_sha256(matched_core),
    }
    contract["binding_sha256"] = _canonical_json_sha256(contract)
    return EmgConsistencyRuntime(
        config=config,
        bundle=bundle,
        spec=spec,
        spec_identity=identity,
        action_index=action_index,
        contract=contract,
    )


def validate_emg_runtime_against_preflight(
    runtime_contract: Mapping[str, Any],
    preflight_contract: Mapping[str, Any],
) -> None:
    """Ensure model compilation consumed exactly the preflight artifact/config."""

    shared = set(preflight_contract) - {"schema_version", "binding_sha256"}
    mismatched = sorted(key for key in shared if runtime_contract.get(key) != preflight_contract.get(key))
    if mismatched:
        raise ValueError(f"EMG runtime compilation differs from artifact preflight fields: {mismatched}")


def curriculum_weight(
    update_index: Any,
    *,
    maximum: float,
    start_update: int,
    ramp_updates: int,
    backend: Any,
) -> Any:
    """JIT-safe linear update curriculum shared by training and validation."""

    update = backend.asarray(update_index, dtype=backend.float32)
    if int(ramp_updates) == 0:
        factor = backend.asarray(update >= int(start_update), dtype=backend.float32)
    else:
        factor = backend.clip(
            (update - float(start_update)) / float(ramp_updates),
            0.0,
            1.0,
        )
    return backend.asarray(maximum, dtype=backend.float32) * factor
