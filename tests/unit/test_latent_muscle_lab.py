from __future__ import annotations

import numpy as np
import pytest

from musclemimic.latent_muscle.action_mask import ActionMask


def _inverse_softplus(value: np.ndarray) -> np.ndarray:
    return np.log(np.expm1(value))


def _require_jax():
    return pytest.importorskip("jax")


# ---------------------------------------------------------------------------
# Pure-numpy tests (always run)
# ---------------------------------------------------------------------------


def test_action_mask_splits_body_and_correction_actions_by_actuator_name():
    mask = ActionMask.from_correction_actuators(
        all_actuator_names=["hip", "ECRL", "shoulder", "FDS2"],
        correction_actuator_names=["ECRL", "FDS2"],
    )

    assert mask.body_actuator_names == ["hip", "shoulder"]
    assert mask.correction_actuator_names == ["ECRL", "FDS2"]

    body, correction = mask.split(np.array([0.1, 0.2, 0.3, 0.4]))
    np.testing.assert_allclose(body, np.array([0.1, 0.3]))
    np.testing.assert_allclose(correction, np.array([0.2, 0.4]))
    np.testing.assert_allclose(mask.merge(body, correction), np.array([0.1, 0.2, 0.3, 0.4]))


def test_action_mask_can_be_built_from_layered_router_partition():
    from environment.overall_environment.src.layered_control import LayeredActuatorRouter

    router = LayeredActuatorRouter(
        all_actuator_names=["hip", "ECRL", "shoulder", "FDS2"],
        body_actuator_names=["hip", "shoulder"],
        grip_actuator_names=["ECRL", "FDS2"],
    )

    mask = ActionMask.from_layered_router(router)

    assert mask.body_actuator_names == router.body_actuator_names
    assert mask.correction_actuator_names == router.grip_actuator_names
    mask.assert_matches_partitions(
        body_actuator_names=router.body_actuator_names,
        correction_actuator_names=router.grip_actuator_names,
    )


# ---------------------------------------------------------------------------
# JAX-dependent tests (skip when jax is not installed)
# ---------------------------------------------------------------------------


def test_latent_action_barrier_maps_raw_residual_inside_state_prior_bounds():
    jax = _require_jax()
    jnp = jax.numpy
    from musclemimic.latent_muscle.lab import latent_action_barrier

    mu = jnp.array([[1.0, -2.0, 0.5]])
    raw_sigma = jnp.asarray(_inverse_softplus(np.array([[0.01, 10.0, 0.5]], dtype=np.float32)))
    raw_latent = jnp.array([[0.0, 20.0, -20.0]])

    z = latent_action_barrier(
        raw_latent=raw_latent,
        prior_mu=mu,
        prior_log_sigma=raw_sigma,
        lambda_lab=2.0,
        sigma_min=0.05,
        sigma_max=1.5,
    )

    expected = np.array([[1.0, 1.0, -0.5]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(z), expected, rtol=1e-5, atol=1e-5)
    upper = np.asarray(mu) + 2.0 * np.array([[0.05, 1.5, 0.5]], dtype=np.float32)
    lower = np.asarray(mu) - 2.0 * np.array([[0.05, 1.5, 0.5]], dtype=np.float32)
    assert np.all(np.asarray(z) <= upper + 1e-6)
    assert np.all(np.asarray(z) >= lower - 1e-6)


def test_lab_action_wrapper_decodes_body_action_and_exposes_latent_diagnostics():
    _require_jax()
    from musclemimic.latent_muscle.lab import LABActionWrapper

    class Prior:
        def __call__(self, state):
            assert state.shape == (3,)
            return np.array([0.5, -0.5]), _inverse_softplus(np.array([0.25, 0.5]))

    class Decoder:
        def __call__(self, state, latent):
            return np.array([state[0] + latent[0], latent[1]])

    wrapper = LABActionWrapper(
        prior=Prior(),
        decoder=Decoder(),
        lambda_lab=1.0,
        sigma_min=0.05,
        sigma_max=1.0,
    )

    output = wrapper(np.array([2.0, 3.0, 4.0]), np.array([20.0, -20.0]), return_info=True)

    np.testing.assert_allclose(output.action, np.array([2.75, -1.0]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(output.latent, np.array([0.75, -1.0]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(output.prior_mu, np.array([0.5, -0.5]))
    np.testing.assert_allclose(output.prior_sigma, np.array([0.25, 0.5]), rtol=1e-5, atol=1e-5)


def test_latent_distillation_loss_combines_action_kl_and_smoothness_terms():
    jax = _require_jax()
    jnp = jax.numpy
    from musclemimic.latent_muscle.losses import latent_distillation_loss

    posterior_mu = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    unit_raw_sigma = jnp.asarray(_inverse_softplus(np.ones((2, 2), dtype=np.float32)))
    posterior_log_sigma = unit_raw_sigma
    prior_mu = jnp.zeros((2, 2))
    prior_log_sigma = unit_raw_sigma
    predicted_action = jnp.array([[0.0, 1.0], [2.0, 3.0]])
    teacher_action = jnp.array([[1.0, 1.0], [0.0, 3.0]])

    losses = latent_distillation_loss(
        predicted_action=predicted_action,
        teacher_action=teacher_action,
        posterior_mu=posterior_mu,
        posterior_log_sigma=posterior_log_sigma,
        prior_mu=prior_mu,
        prior_log_sigma=prior_log_sigma,
        previous_predicted_action=jnp.array([[0.0, 0.5], [1.0, 2.0]]),
        action_weight=1.0,
        kl_weight=0.5,
        smooth_weight=0.25,
    )

    assert np.isclose(float(losses["action_mse"]), 1.25)
    assert np.isclose(float(losses["kl"]), 0.25)
    assert np.isclose(float(losses["smooth_mse"]), 0.5625)
    assert np.isclose(float(losses["total_loss"]), 1.515625)


def test_latent_networks_return_expected_distribution_and_action_shapes():
    jax = _require_jax()
    jnp = jax.numpy
    from musclemimic.latent_muscle.networks import ConditionalPrior, LatentDecoder, PosteriorEncoder

    state = jnp.ones((4, 6), dtype=jnp.float32)
    reference = jnp.ones((4, 5), dtype=jnp.float32) * 2.0
    latent = jnp.ones((4, 3), dtype=jnp.float32)

    posterior = PosteriorEncoder(latent_dim=3, hidden_layer_dims=(8, 8))
    prior = ConditionalPrior(latent_dim=3, hidden_layer_dims=(8, 8))
    decoder = LatentDecoder(action_dim=7, hidden_layer_dims=(8, 8), bounded_action=True)

    rng = jax.random.PRNGKey(0)
    post_vars = posterior.init(rng, state, reference)
    prior_vars = prior.init(rng, state)
    dec_vars = decoder.init(rng, state, latent)

    post_mu, post_log_sigma = posterior.apply(post_vars, state, reference)
    prior_mu, prior_log_sigma = prior.apply(prior_vars, state)
    action = decoder.apply(dec_vars, state, latent)

    assert post_mu.shape == (4, 3)
    assert post_log_sigma.shape == (4, 3)
    assert prior_mu.shape == (4, 3)
    assert prior_log_sigma.shape == (4, 3)
    assert action.shape == (4, 7)
    assert float(jnp.min(action)) >= -1.0
    assert float(jnp.max(action)) <= 1.0


def test_bounded_latent_decoder_uses_symmetric_muscle_action_range():
    jax = _require_jax()
    jnp = jax.numpy
    from flax.core import freeze, unfreeze
    from musclemimic.latent_muscle.networks import LatentDecoder

    state = jnp.ones((1, 2), dtype=jnp.float32)
    latent = jnp.ones((1, 1), dtype=jnp.float32)
    decoder = LatentDecoder(action_dim=1, hidden_layer_dims=(), bounded_action=True)
    variables = decoder.init(jax.random.PRNGKey(1), state, latent)
    params = unfreeze(variables)
    params["params"]["action"]["kernel"] = jnp.zeros_like(params["params"]["action"]["kernel"])
    params["params"]["action"]["bias"] = jnp.array([-20.0], dtype=jnp.float32)

    action = decoder.apply(freeze(params), state, latent)

    assert float(action[0, 0]) < -0.99
