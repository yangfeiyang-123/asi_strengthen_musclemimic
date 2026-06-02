"""Loss helpers for behavior cloning policy distillation."""

from __future__ import annotations

import jax.numpy as jnp


def distribution_mean(pi):
    """Return the mean/mode tensor from a policy distribution."""
    mean_attr = getattr(pi, "mean", None)
    if callable(mean_attr):
        return mean_attr()
    if mean_attr is not None:
        return mean_attr
    mode_attr = getattr(pi, "mode", None)
    if callable(mode_attr):
        return mode_attr()
    raise TypeError(f"Unsupported policy distribution type: {type(pi)!r}")


def bc_loss(
    *,
    student_mu,
    teacher_action,
    student_value=None,
    teacher_value=None,
    action_mse_weight: float = 1.0,
    value_distill_weight: float = 0.0,
) -> dict[str, jnp.ndarray]:
    """Compute action MSE plus optional value distillation loss."""
    action_mse = jnp.mean(jnp.square(student_mu - teacher_action))
    if teacher_value is None or student_value is None or value_distill_weight == 0.0:
        value_mse = jnp.asarray(0.0, dtype=action_mse.dtype)
    else:
        value_mse = jnp.mean(jnp.square(student_value - teacher_value))
    total = action_mse_weight * action_mse + value_distill_weight * value_mse
    return {
        "total_loss": total,
        "action_mse": action_mse,
        "value_mse": value_mse,
    }
