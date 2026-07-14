"""Physical muscle-signal contracts used by distillation and synergy fitting.

The policy action is normalized, while MuJoCo ``data.ctrl`` follows the
ordered actuator control range and may be signed.  The two are deliberately
kept separate: raw applied control is evidence, and a non-negative unit
excitation is an explicitly named affine transform suitable for NMF.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

PHYSICAL_SIGNAL_SCHEMA_VERSION = "physical_muscle_transition_v1"
UNIT_EXCITATION_TRANSFORM = "ordered_ctrlrange_affine_to_unit_interval_v1"
MUSCLE_EXCITATION_SOURCE = "teacher_ctrl_physical"
MUSCLE_EXCITATION_SEMANTICS = "unit_interval_excitation"
MUSCLE_ACTIVATION_SOURCE = "transition_state.data.act via model.actuator_actadr"
MUSCLE_ACTIVATION_SEMANTICS = "mujoco_unit_interval_activation_state"
UNIT_INTERVAL_ROUNDOFF_POLICY = "fail_outside_unit_interval_then_clamp_within_tolerance_only"
UNIT_INTERVAL_TOLERANCE = 1e-6


def validate_ordered_ctrlrange(actuator_names: Sequence[str], actuator_ctrlrange: Any) -> np.ndarray:
    names = [str(name) for name in actuator_names]
    if not names or len(set(names)) != len(names):
        raise ValueError("actuator_names must be non-empty and unique")
    limits = np.asarray(actuator_ctrlrange, dtype=np.float64)
    if limits.shape != (len(names), 2):
        raise ValueError(f"actuator_ctrlrange must have shape ({len(names)}, 2), got {limits.shape}")
    if not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("actuator_ctrlrange must contain finite increasing bounds")
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


def physical_ctrl_to_unit_excitation(
    ctrl: Any,
    actuator_ctrlrange: Any,
) -> np.ndarray:
    """Map raw applied ctrl to [0, 1] without silently clipping evidence."""
    normalized = physical_ctrl_to_normalized_action(ctrl, actuator_ctrlrange, clip=False)
    unit = 0.5 * (normalized + 1.0)
    if np.any(unit < -1e-6) or np.any(unit > 1.0 + 1e-6):
        raise ValueError("unit excitation transform produced values outside [0, 1]")
    return unit.astype(np.float32)


def unit_excitation_to_physical_ctrl(
    excitation: Any,
    actuator_ctrlrange: Any,
) -> np.ndarray:
    value = np.asarray(excitation, dtype=np.float64)
    if not np.all(np.isfinite(value)) or np.any(value < -1e-6) or np.any(value > 1.0 + 1e-6):
        raise ValueError("unit excitation must be finite and lie in [0, 1]")
    return normalized_action_to_physical_ctrl(2.0 * value - 1.0, actuator_ctrlrange)


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
        "nonnegative": True,
    }
    if any(excitation.get(key) != value for key, value in expected_excitation.items()):
        raise ValueError(
            "physical signal contract requires exact excitation/activation semantics; "
            "muscle_excitation semantics must declare the exact unit-interval "
            "ordered-ctrlrange transform contract"
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
