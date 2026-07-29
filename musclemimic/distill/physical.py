"""Physical muscle-signal contracts used by distillation and synergy fitting.

Raw MuJoCo ``data.ctrl`` is retained as evidence.  Production muscle
excitation is *not* a coordinate normalization: after every channel is proven
to be a scalar MuJoCo muscle actuator, its effective excitation is exactly
``clip(raw_ctrl, 0, 1)``.  This mirrors MuJoCo's muscle control semantics and
prevents signed actuator coordinates from being relabeled as physiology.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

PHYSICAL_SIGNAL_SCHEMA_VERSION = "physical_muscle_transition_v2"
PHYSICAL_CAPTURE_SCHEMA_VERSION = "physical_capture_spec_v2"
MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION = "verified_mujoco_muscle_channels_v2"
UNIT_EXCITATION_TRANSFORM = "verified_muscle_raw_ctrl_clip_to_unit_interval_v2"
MUSCLE_EXCITATION_SOURCE = "teacher_ctrl_physical"
MUSCLE_EXCITATION_SEMANTICS = "mujoco_effective_muscle_excitation"
MUSCLE_EXCITATION_FORMULA = "clip(raw_ctrl,0,1)"
MUSCLE_EXCITATION_ROUNDOFF_POLICY = "exact_clip_to_closed_mujoco_muscle_control_domain"
MUSCLE_ACTIVATION_SOURCE = "transition_state.data.act via model.actuator_actadr"
MUSCLE_ACTIVATION_SEMANTICS = "mujoco_unit_interval_activation_state"
UNIT_INTERVAL_ROUNDOFF_POLICY = "fail_outside_unit_interval_then_clamp_within_tolerance_only"
UNIT_INTERVAL_TOLERANCE = 1e-6


@dataclass(frozen=True)
class MuscleChannelContract:
    """Name-aligned proof that every reported channel is a scalar muscle."""

    actuator_names: tuple[str, ...]
    actuator_ids: tuple[int, ...]
    actuator_dyntype: tuple[str, ...]
    actuator_actnum: tuple[int, ...]
    actuator_actadr: tuple[int, ...]
    model_na: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
            "actuator_names": list(self.actuator_names),
            "actuator_ids": list(self.actuator_ids),
            "actuator_dyntype": list(self.actuator_dyntype),
            "actuator_actnum": list(self.actuator_actnum),
            "actuator_actadr": list(self.actuator_actadr),
            "model_na": int(self.model_na),
        }

    def subset(self, indices: Sequence[int]) -> MuscleChannelContract:
        selected = tuple(int(index) for index in indices)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or min(selected) < 0
            or max(selected) >= len(self.actuator_names)
        ):
            raise ValueError("muscle channel subset indices are invalid")
        return MuscleChannelContract(
            actuator_names=tuple(self.actuator_names[index] for index in selected),
            actuator_ids=tuple(self.actuator_ids[index] for index in selected),
            actuator_dyntype=tuple(self.actuator_dyntype[index] for index in selected),
            actuator_actnum=tuple(self.actuator_actnum[index] for index in selected),
            actuator_actadr=tuple(self.actuator_actadr[index] for index in selected),
            model_na=self.model_na,
        )


def resolve_muscle_channel_contract(
    model: Any,
    actuator_names: Sequence[str],
) -> MuscleChannelContract:
    """Resolve and verify muscle ``dyntype``, ``actnum`` and ``actadr``."""

    import mujoco

    names = _validated_names(actuator_names)
    actuator_ids: list[int] = []
    actuator_dyntype: list[str] = []
    actuator_actnum: list[int] = []
    actuator_actadr: list[int] = []
    model_na = int(model.na)
    for name in names:
        actuator_id = int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )
        )
        if actuator_id < 0:
            raise ValueError(f"muscle channel actuator {name!r} is absent from the MuJoCo model")
        dyntype = int(model.actuator_dyntype[actuator_id])
        actnum = int(model.actuator_actnum[actuator_id])
        actadr = int(model.actuator_actadr[actuator_id])
        if dyntype != int(mujoco.mjtDyn.mjDYN_MUSCLE):
            raise ValueError(
                f"actuator {name!r} has dyntype={dyntype}; production excitation requires dyntype=muscle"
            )
        if actnum != 1 or actadr < 0 or actadr >= model_na:
            raise ValueError(
                "production muscle signals require one addressable activation state per channel; "
                f"{name!r} has actnum={actnum}, actadr={actadr}, model.na={model_na}"
            )
        actuator_ids.append(actuator_id)
        actuator_dyntype.append("muscle")
        actuator_actnum.append(actnum)
        actuator_actadr.append(actadr)
    if len(set(actuator_ids)) != len(actuator_ids):
        raise ValueError("muscle channel actuator ids are not unique")
    if len(set(actuator_actadr)) != len(actuator_actadr):
        raise ValueError("muscle channel activation addresses are not unique")
    return MuscleChannelContract(
        actuator_names=names,
        actuator_ids=tuple(actuator_ids),
        actuator_dyntype=tuple(actuator_dyntype),
        actuator_actnum=tuple(actuator_actnum),
        actuator_actadr=tuple(actuator_actadr),
        model_na=model_na,
    )


def validate_muscle_channel_contract(
    payload: MuscleChannelContract | Mapping[str, Any],
    *,
    expected_names: Sequence[str] | None = None,
) -> MuscleChannelContract:
    """Parse a persisted v2 channel proof and reject legacy/partial records."""

    if isinstance(payload, MuscleChannelContract):
        contract = payload
    else:
        if not isinstance(payload, Mapping):
            raise ValueError("muscle_channel_contract must be a v2 object")
        if payload.get("schema_version") != MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION:
            raise ValueError(
                "muscle_channel_contract schema is unsupported; legacy channel metadata "
                f"is rejected (expected {MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION!r})"
            )
        names = _validated_names(payload.get("actuator_names", ()))
        width = len(names)
        actuator_ids = _integer_vector(payload.get("actuator_ids"), width, "actuator_ids")
        actuator_actnum = _integer_vector(payload.get("actuator_actnum"), width, "actuator_actnum")
        actuator_actadr = _integer_vector(payload.get("actuator_actadr"), width, "actuator_actadr")
        dyntype = tuple(str(value) for value in payload.get("actuator_dyntype", ()))
        if len(dyntype) != width or any(value != "muscle" for value in dyntype):
            raise ValueError("production excitation requires every actuator_dyntype entry to be 'muscle'")
        if len(set(actuator_ids)) != width or any(value < 0 for value in actuator_ids):
            raise ValueError("muscle channel actuator_ids must be unique and non-negative")
        if any(value != 1 for value in actuator_actnum):
            raise ValueError("production muscle signals require actuator_actnum=1 for every channel")
        model_na = _strict_nonnegative_int(payload.get("model_na"), "model_na")
        if model_na <= 0:
            raise ValueError("muscle channel model_na must be positive")
        if (
            len(set(actuator_actadr)) != width
            or any(value < 0 or value >= model_na for value in actuator_actadr)
        ):
            raise ValueError("muscle channel actuator_actadr must be unique and lie in [0, model_na)")
        contract = MuscleChannelContract(
            actuator_names=names,
            actuator_ids=actuator_ids,
            actuator_dyntype=dyntype,
            actuator_actnum=actuator_actnum,
            actuator_actadr=actuator_actadr,
            model_na=model_na,
        )
    if expected_names is not None and contract.actuator_names != tuple(str(name) for name in expected_names):
        raise ValueError("muscle channel contract actuator_names/order mismatch")
    return contract


def validate_ordered_ctrlrange(actuator_names: Sequence[str], actuator_ctrlrange: Any) -> np.ndarray:
    names = _validated_names(actuator_names)
    limits = np.asarray(actuator_ctrlrange, dtype=np.float64)
    if limits.shape != (len(names), 2):
        raise ValueError(f"actuator_ctrlrange must have shape ({len(names)}, 2), got {limits.shape}")
    if not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("actuator_ctrlrange must contain finite increasing bounds")
    return limits


def validate_unit_muscle_ctrlrange(
    actuator_names: Sequence[str],
    actuator_ctrlrange: Any,
) -> np.ndarray:
    """Require the v2 physical muscle ABI ``ctrlrange=[0,1]`` exactly."""

    limits = validate_ordered_ctrlrange(actuator_names, actuator_ctrlrange)
    expected = np.broadcast_to(np.asarray([0.0, 1.0]), limits.shape)
    if not np.array_equal(limits, expected):
        raise ValueError(
            "production muscle signal v2 requires every verified muscle ctrlrange to be exactly [0,1]; "
            "legacy signed/mixed control ranges are rejected"
        )
    return limits


def normalized_action_to_physical_ctrl(
    action: Any,
    actuator_ctrlrange: Any,
    *,
    clip: bool = False,
) -> np.ndarray:
    value = np.asarray(action, dtype=np.float64)
    limits = np.asarray(actuator_ctrlrange, dtype=np.float64)
    if limits.ndim != 2 or limits.shape[-1] != 2 or value.shape[-1] != limits.shape[0]:
        raise ValueError("action and ctrlrange dimensions do not agree")
    if not np.all(np.isfinite(value)):
        raise ValueError("normalized action contains non-finite values")
    if not clip and np.any(np.abs(value) > 1.0 + 1e-6):
        raise ValueError("normalized action lies outside [-1, 1]")
    bounded = np.clip(value, -1.0, 1.0) if clip else value
    return limits[:, 0] + 0.5 * (bounded + 1.0) * (limits[:, 1] - limits[:, 0])


def physical_ctrl_to_normalized_action(
    ctrl: Any,
    actuator_ctrlrange: Any,
    *,
    clip: bool = False,
) -> np.ndarray:
    value = np.asarray(ctrl, dtype=np.float64)
    limits = np.asarray(actuator_ctrlrange, dtype=np.float64)
    if limits.ndim != 2 or limits.shape[-1] != 2 or value.shape[-1] != limits.shape[0]:
        raise ValueError("ctrl and ctrlrange dimensions do not agree")
    if not np.all(np.isfinite(value)):
        raise ValueError("physical ctrl contains non-finite values")
    tolerance = 1e-6
    if not clip and (np.any(value < limits[:, 0] - tolerance) or np.any(value > limits[:, 1] + tolerance)):
        raise ValueError("physical ctrl lies outside the ordered actuator ctrlrange")
    bounded = np.clip(value, limits[:, 0], limits[:, 1]) if clip else value
    return 2.0 * (bounded - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1.0


def physical_ctrl_to_effective_muscle_excitation(
    ctrl: Any,
    *,
    channel_contract: MuscleChannelContract | Mapping[str, Any],
) -> np.ndarray:
    """Return MuJoCo effective muscle excitation from retained raw ctrl.

    Clipping here is the declared MuJoCo muscle semantic, not error recovery.
    The raw input must therefore remain persisted separately.
    """

    contract = validate_muscle_channel_contract(channel_contract)
    value = np.asarray(ctrl, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != len(contract.actuator_names):
        raise ValueError(
            "raw physical ctrl width does not match the verified muscle channel contract"
        )
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("raw physical ctrl must be a non-empty finite array")
    return np.clip(value, 0.0, 1.0).astype(np.float32)


def physical_ctrl_to_unit_excitation(
    ctrl: Any,
    *,
    channel_contract: MuscleChannelContract | Mapping[str, Any],
) -> np.ndarray:
    """Deprecated compatibility alias for the explicit v2 API."""

    import warnings

    warnings.warn(
        "physical_ctrl_to_unit_excitation is deprecated; use "
        "physical_ctrl_to_effective_muscle_excitation",
        DeprecationWarning,
        stacklevel=2,
    )
    return physical_ctrl_to_effective_muscle_excitation(
        ctrl,
        channel_contract=channel_contract,
    )


def effective_muscle_excitation_to_physical_ctrl(
    excitation: Any,
    actuator_ctrlrange: Any,
    *,
    actuator_names: Sequence[str],
    channel_contract: MuscleChannelContract | Mapping[str, Any],
) -> np.ndarray:
    """Validate effective muscle excitation for the physical muscle ABI."""

    contract = validate_muscle_channel_contract(
        channel_contract,
        expected_names=actuator_names,
    )
    validate_unit_muscle_ctrlrange(contract.actuator_names, actuator_ctrlrange)
    value = np.asarray(excitation, dtype=np.float64)
    if not np.all(np.isfinite(value)) or np.any(value < -1e-6) or np.any(value > 1.0 + 1e-6):
        raise ValueError("unit excitation must be finite and lie in [0, 1]")
    if value.ndim < 1 or value.shape[-1] != len(contract.actuator_names):
        raise ValueError("unit excitation width differs from the muscle channel contract")
    return np.clip(value, 0.0, 1.0)


def unit_excitation_to_physical_ctrl(
    excitation: Any,
    actuator_ctrlrange: Any,
    *,
    actuator_names: Sequence[str],
    channel_contract: MuscleChannelContract | Mapping[str, Any],
) -> np.ndarray:
    """Deprecated compatibility alias for the explicit inverse v2 API."""

    import warnings

    warnings.warn(
        "unit_excitation_to_physical_ctrl is deprecated; use "
        "effective_muscle_excitation_to_physical_ctrl",
        DeprecationWarning,
        stacklevel=2,
    )
    return effective_muscle_excitation_to_physical_ctrl(
        excitation,
        actuator_ctrlrange,
        actuator_names=actuator_names,
        channel_contract=channel_contract,
    )


def validate_unit_muscle_excitation(
    excitation: Any,
    *,
    tolerance: float = UNIT_INTERVAL_TOLERANCE,
) -> np.ndarray:
    """Validate explicitly declared physical excitation in the unit interval.

    A signed normalized policy action or raw MuJoCo control is not accepted as
    excitation.  Only floating-point roundoff within ``tolerance`` is clipped.
    """

    value = np.asarray(excitation, dtype=np.float64)
    tol = float(tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("excitation unit-interval tolerance must be finite and non-negative")
    if value.ndim < 1 or value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("muscle_excitation must be a non-empty finite array")
    if np.any(value < -tol) or np.any(value > 1.0 + tol):
        raise ValueError(
            "muscle_excitation lies outside [0,1] beyond numeric tolerance; "
            "signed normalized action/raw ctrl must not be relabeled as physical excitation"
        )
    return np.clip(value, 0.0, 1.0).astype(np.float32)


def validate_unit_muscle_activation(
    activation: Any,
    *,
    tolerance: float = UNIT_INTERVAL_TOLERANCE,
) -> np.ndarray:
    """Validate MuJoCo activation state without relabeling excitation/control.

    Only floating-point roundoff within ``tolerance`` is clipped.  Materially
    signed or supra-unit values fail closed because they cannot be interpreted
    as the unit activation state declared by this data contract.
    """

    value = np.asarray(activation, dtype=np.float64)
    tol = float(tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("activation unit-interval tolerance must be finite and non-negative")
    if value.ndim < 1 or value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("muscle_activation must be a non-empty finite array")
    if np.any(value < -tol) or np.any(value > 1.0 + tol):
        raise ValueError(
            "muscle_activation lies outside [0,1] beyond numeric tolerance; "
            "excitation/signed ctrl must not be relabeled as activation"
        )
    return np.clip(value, 0.0, 1.0).astype(np.float32)


def validate_physical_signal_semantics(payload: Any) -> dict[str, Any]:
    """Validate the exact persisted excitation/activation semantics record."""

    if not isinstance(payload, dict):
        raise ValueError("physical_signal_semantics must be an object")
    if payload.get("schema_version") != PHYSICAL_SIGNAL_SCHEMA_VERSION:
        raise ValueError(f"physical_signal_semantics.schema_version must be {PHYSICAL_SIGNAL_SCHEMA_VERSION!r}")
    excitation = payload.get("muscle_excitation")
    if not isinstance(excitation, dict):
        raise ValueError("physical_signal_semantics lacks muscle_excitation")
    expected_excitation = {
        "source": MUSCLE_EXCITATION_SOURCE,
        "semantics": MUSCLE_EXCITATION_SEMANTICS,
        "transform": UNIT_EXCITATION_TRANSFORM,
        "formula": MUSCLE_EXCITATION_FORMULA,
        "roundoff_policy": MUSCLE_EXCITATION_ROUNDOFF_POLICY,
        "nonnegative": True,
    }
    if any(excitation.get(key) != value for key, value in expected_excitation.items()):
        raise ValueError(
            "physical signal contract requires exact excitation/activation semantics; "
            "muscle_excitation semantics must declare the exact verified-muscle "
            "raw-ctrl clipping contract"
        )
    activation = payload.get("muscle_activation")
    if not isinstance(activation, dict):
        raise ValueError("physical_signal_semantics lacks muscle_activation")
    expected = {
        "source": MUSCLE_ACTIVATION_SOURCE,
        "semantics": MUSCLE_ACTIVATION_SEMANTICS,
        "nonnegative": True,
        "upper_bound": 1.0,
        "roundoff_policy": UNIT_INTERVAL_ROUNDOFF_POLICY,
    }
    if any(activation.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "muscle_activation semantics must declare the exact unit-interval transition_state.data.act source contract"
        )
    return payload


def validate_activation_valid_mask(
    values: Any,
    *,
    expected_width: int,
) -> np.ndarray:
    """Validate the name-aligned mask produced from ``actuator_actadr``."""

    mask = np.asarray(values)
    if mask.shape != (int(expected_width),) or mask.dtype.kind != "b":
        raise ValueError(
            "physical_capture.activation_valid_mask must be boolean and match "
            f"the ordered actuator width ({int(expected_width)})"
        )
    return mask.astype(bool, copy=False)


def physical_signal_metadata() -> dict[str, Any]:
    return {
        "schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
        "teacher_ctrl_physical": {
            "source": "transition_state.data.ctrl",
            "semantics": "raw_applied_mujoco_ctrl_ordered_by_actuator_names",
            "nonnegative": False,
        },
        "muscle_excitation": {
            "source": MUSCLE_EXCITATION_SOURCE,
            "semantics": MUSCLE_EXCITATION_SEMANTICS,
            "transform": UNIT_EXCITATION_TRANSFORM,
            "formula": MUSCLE_EXCITATION_FORMULA,
            "roundoff_policy": MUSCLE_EXCITATION_ROUNDOFF_POLICY,
            "nonnegative": True,
        },
        "muscle_activation": {
            "source": MUSCLE_ACTIVATION_SOURCE,
            "semantics": MUSCLE_ACTIVATION_SEMANTICS,
            "nonnegative": True,
            "upper_bound": 1.0,
            "roundoff_policy": UNIT_INTERVAL_ROUNDOFF_POLICY,
            "validity_mask": "physical_capture.activation_valid_mask",
        },
    }


def _validated_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in values)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("actuator_names must be non-empty and unique")
    return names


def _integer_vector(value: Any, width: int, field: str) -> tuple[int, ...]:
    array = np.asarray(value)
    if (
        array.shape != (int(width),)
        or np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(f"muscle channel {field} must be an integer vector of width {int(width)}")
    return tuple(int(item) for item in array.tolist())


def _strict_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"muscle channel {field} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"muscle channel {field} must be non-negative")
    return result
