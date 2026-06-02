"""Tests for behavior cloning distillation losses."""

import jax.numpy as jnp
import numpy as np
import distrax

from musclemimic.distill.losses import bc_loss, distribution_mean


def test_distribution_mean_supports_distrax_multivariate_normal_diag():
    pi = distrax.MultivariateNormalDiag(
        loc=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        scale_diag=jnp.ones((2, 2)),
    )

    np.testing.assert_allclose(distribution_mean(pi), np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_bc_loss_uses_action_mse_and_optional_value_distill():
    student_mu = jnp.array([[0.0, 1.0], [2.0, 3.0]])
    teacher_action = jnp.array([[1.0, 1.0], [0.0, 3.0]])
    student_value = jnp.array([1.0, 4.0])
    teacher_value = jnp.array([3.0, 2.0])

    losses = bc_loss(
        student_mu=student_mu,
        teacher_action=teacher_action,
        student_value=student_value,
        teacher_value=teacher_value,
        action_mse_weight=1.0,
        value_distill_weight=0.5,
    )

    assert np.isclose(float(losses["action_mse"]), 1.25)
    assert np.isclose(float(losses["value_mse"]), 4.0)
    assert np.isclose(float(losses["total_loss"]), 3.25)


def test_bc_loss_without_teacher_value_has_zero_value_mse():
    losses = bc_loss(
        student_mu=jnp.zeros((2, 3)),
        teacher_action=jnp.ones((2, 3)),
        student_value=jnp.zeros(2),
        teacher_value=None,
    )

    assert np.isclose(float(losses["action_mse"]), 1.0)
    assert np.isclose(float(losses["value_mse"]), 0.0)
    assert np.isclose(float(losses["total_loss"]), 1.0)
