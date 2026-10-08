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


def distribution_log_std(pi):
    """Return diagonal Gaussian log std when exposed by a policy distribution."""
    scale = getattr(pi, "scale_diag", None)
    if callable(scale):
        scale = scale()
    if scale is None:
        stddev = getattr(pi, "stddev", None)
        scale = stddev() if callable(stddev) else stddev
    if scale is None:
        raise TypeError(f"Unsupported policy distribution std type: {type(pi)!r}")
    return jnp.log(jnp.asarray(scale))


def gaussian_diag_kl(teacher_mu, teacher_log_std, student_mu, student_log_std):
    """Mean KL(N_teacher || N_student) for diagonal Gaussian policies."""
    teacher_var = jnp.exp(2.0 * teacher_log_std)
    student_var = jnp.exp(2.0 * student_log_std)
    per_dim = (
        student_log_std
        - teacher_log_std
        + (teacher_var + jnp.square(teacher_mu - student_mu)) / (2.0 * student_var)
        - 0.5
    )
    return jnp.mean(jnp.sum(per_dim, axis=-1))


def bc_loss(
    *,
    student_mu,
    teacher_action,
    student_value=None,
    teacher_value=None,
    student_log_std=None,
    teacher_mu=None,
    teacher_log_std=None,
    action_mse_weight: float = 1.0,
    value_distill_weight: float = 0.0,
    gaussian_kl_weight: float = 0.0,
) -> dict[str, jnp.ndarray]:
    """Compute action MSE plus optional value distillation loss."""
    action_mse = jnp.mean(jnp.square(student_mu - teacher_action))
    if teacher_value is None or student_value is None or value_distill_weight == 0.0:
        value_mse = jnp.asarray(0.0, dtype=action_mse.dtype)
    else:
        value_mse = jnp.mean(jnp.square(student_value - teacher_value))
    if (
        gaussian_kl_weight
        and teacher_mu is not None
        and teacher_log_std is not None
        and student_log_std is not None
    ):
        gaussian_kl = gaussian_diag_kl(teacher_mu, teacher_log_std, student_mu, student_log_std)
    else:
        gaussian_kl = jnp.asarray(0.0, dtype=action_mse.dtype)
    total = action_mse_weight * action_mse + value_distill_weight * value_mse + gaussian_kl_weight * gaussian_kl
    return {
        "total_loss": total,
        "action_mse": action_mse,
        "value_mse": value_mse,
        "gaussian_kl": gaussian_kl,
    }
