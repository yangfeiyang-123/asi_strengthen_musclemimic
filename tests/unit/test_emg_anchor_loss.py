"""Tests for the JAX-side EMG anchoring loss.

Two properties carry the paper claim.  First, the loss is *zero* inside human
trial-to-trial variability, so the term anchors coordination without dictating
an exact activation.  Second, comparison happens strictly in the measured
subspace via ``y = P a`` — the 354-wide simulated action is never asked to match
a signal nobody recorded.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from musclemimic.physiology.emg_anchor import (  # noqa: E402
    DEFAULT_HUBER_DELTA,
    DEFAULT_TUBE_KAPPA,
    EMG_ANCHOR_SIGNAL,
    build_emg_anchor_spec,
    build_emg_observation_projection,
    emg_anchor_metrics,
    emg_synergy_metrics,
    phase_bin_index,
    project_ordered_activation,
    tube_distance,
)
from musclemimic.physiology.emg_reference import (  # noqa: E402
    EMG_SYNERGY_PROJECTION_METHOD,
    EMG_SYNERGY_RIDGE,
    build_emg_dual_track_normalization,
    build_phase_reference_tube,
)

ACTUATORS = ("delt_ant_r", "delt_med_r", "bic_l_r", "tric_lat_r", "fcu_r", "soleus_r")
CHANNELS = ("deltoid_anterior", "biceps_brachii", "triceps_lateral")
PHASE_BINS = 4


def _mapping(exclude_last=True):
    channels = [
        {
            "emg_channel": "deltoid_anterior",
            "mapping_status": "verified",
            "simulation_actuators": ["delt_ant_r", "delt_med_r"],
            "weights": [0.7, 0.3],
        },
        {
            "emg_channel": "biceps_brachii",
            "mapping_status": "verified",
            "simulation_actuators": ["bic_l_r"],
            "weights": [1.0],
        },
        {
            "emg_channel": "triceps_lateral",
            "mapping_status": "verified",
            "simulation_actuators": ["tric_lat_r"],
            "weights": [1.0],
        },
    ]
    if exclude_last:
        channels.append(
            {
                "emg_channel": "tibialis_anterior",
                "mapping_status": "excluded_no_verified_model_homolog",
                "simulation_actuators": ["soleus_r"],
                "weights": [1.0],
            }
        )
    return {
        "mapping_id": "test_mapping_v1",
        "mapping_sha256": "a" * 64,
        "mapping_review_status": "verified",
        "channels": channels,
    }


def _envelopes(num_trials=6, num_samples=32, *, level=0.5):
    rng = np.random.default_rng(0)
    values = np.full((num_trials, num_samples, len(CHANNELS)), float(level))
    values += 0.01 * rng.standard_normal(values.shape)
    return np.clip(values, 0.0, 1.0)


def _reviewed_provenance(num_trials: int):
    return {
        "subject": "P002",
        "session": "2025-07-11",
        "normalization": "mvc_percent",
        "review_evidence": [{"reviewer": "expert", "date": "2025-07-20"}],
        "trial_qc_review": {
            "schema_version": "emg_trial_channel_qc_review_v1",
            "action": "forehand_clear",
            "review_status": "verified",
            "training_enabled": True,
            "source_path": "/controlled/evidence/emg_trial_qc_review.json",
            "review_sha256": "d" * 64,
            "mapping_sha256": "a" * 64,
            "reviewer_id": "domain_expert",
            "reviewed_at": "2025-07-20T00:00:00Z",
            "review_evidence": ["controlled-review-record"],
            "trial_decisions": [
                {
                    "trial_id": f"trial_{index:03d}",
                    "decision": "include",
                    "reason": "reviewed synthetic fixture",
                    "mvc_normalized_emg_sha256": f"{index + 1:064x}",
                    "preprocessing_qc_sha256": f"{index + 101:064x}",
                }
                for index in range(num_trials)
            ],
            "channel_decisions": [
                {
                    "emg_channel": name,
                    "decision": "include_after_review",
                    "reason": "reviewed synthetic fixture",
                }
                for name in CHANNELS
            ],
            "risk_decisions": [
                {
                    "risk_id": risk_id,
                    "decision": "mitigated",
                    "reason": "reviewed synthetic fixture",
                    "evidence": ["controlled-review-record"],
                }
                for risk_id in ("s9_progressive_near_flatline", "super_mvc")
            ],
        },
    }


def _tube(**overrides):
    basis = np.stack(
        [np.array([1.0, 0.6, 0.1]), np.array([0.1, 0.2, 1.0])],
        axis=1,
    )
    payload = {
        "reference_id": "anchor_test",
        "action_envelopes": {"forehand_clear": _envelopes()},
        "channel_names": CHANNELS,
        "synergy_basis": basis,
        "mapping_binding": {
            "mapping_id": "test_mapping_v1",
            "mapping_sha256": "a" * 64,
            "mapping_review_status": "verified",
            "acquired_channel_count": 4,
            "comparable_channel_count": len(CHANNELS),
            "actuator_schema_hash": "b" * 64,
        },
        "synergy_binding": {
            "basis_id": "test_nmf_k2",
            "basis_sha256": "c" * 64,
            "synergy_count": 2,
            "channel_normalization": "unit_variance_per_channel",
            "projection_method": EMG_SYNERGY_PROJECTION_METHOD,
            "projection_ridge": EMG_SYNERGY_RIDGE,
        },
        "phase_bin_count": PHASE_BINS,
        "review_status": "verified",
        "training_enabled": True,
    }
    payload.update(overrides)
    if "provenance" not in overrides:
        num_trials = int(payload["action_envelopes"]["forehand_clear"].shape[0])
        payload["provenance"] = _reviewed_provenance(num_trials)
    if "normalization_binding" not in payload:
        values = payload["action_envelopes"]["forehand_clear"]
        normalization_samples = [
            np.concatenate(
                [values[index], np.ones((2, values.shape[2]), dtype=np.float64)],
                axis=0,
            )
            for index in range(values.shape[0])
        ]
        payload["normalization_binding"] = build_emg_dual_track_normalization(
            action_samples={"forehand_clear": normalization_samples},
            channel_names=CHANNELS,
            training_cohorts={
                "forehand_clear": [
                    {
                        "trial_id": f"trial_{index:03d}",
                        "mvc_normalized_emg_sha256": f"{index + 1:064x}",
                    }
                    for index in range(values.shape[0])
                ]
            },
            mvc_final_reference_mv=np.ones(len(CHANNELS)),
            mvc_reference_binding={
                "path": "/controlled/preprocessing_mvc_reference.json",
                "sha256": "f" * 64,
                "scope": "participant",
                "algorithm": "controlled fixture MVC",
            },
        )
    return build_phase_reference_tube(**payload)


def _spec(**overrides):
    kwargs = {
        "actuator_names": ACTUATORS,
        "activation_addresses": np.arange(len(ACTUATORS), dtype=np.int32),
        "muscle_channel_core_fingerprint": "d" * 64,
    }
    kwargs.update(overrides)
    return build_emg_anchor_spec(_tube(), _mapping(), **kwargs)


def _activation(level=0.5):
    """An ordered activation whose projection sits at the tube centre."""

    return jnp.full((len(ACTUATORS),), float(level), dtype=jnp.float32)


# ---------------------------------------------------------------- projection


def test_projection_keeps_only_comparable_channels():
    projection, names = build_emg_observation_projection(_mapping(), ACTUATORS)

    # The excluded electrode has no verified model homolog, so it must not
    # appear as a row — otherwise the loss invents a comparison.
    assert names == list(CHANNELS)
    assert projection.shape == (len(CHANNELS), len(ACTUATORS))
    assert projection[:, ACTUATORS.index("soleus_r")].tolist() == [0.0, 0.0, 0.0]


def test_projection_rows_are_convex_weightings():
    projection, _ = build_emg_observation_projection(_mapping(), ACTUATORS)

    np.testing.assert_allclose(projection.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(projection >= 0.0)


def test_projection_places_weights_by_actuator_name_not_position():
    projection, _ = build_emg_observation_projection(_mapping(), ACTUATORS)

    row = projection[0]
    assert row[ACTUATORS.index("delt_ant_r")] == pytest.approx(0.7)
    assert row[ACTUATORS.index("delt_med_r")] == pytest.approx(0.3)


def test_reordering_actuators_moves_weights_with_the_names():
    shuffled = ("soleus_r", "bic_l_r", "delt_med_r", "fcu_r", "tric_lat_r", "delt_ant_r")
    projection, _ = build_emg_observation_projection(_mapping(), shuffled)

    row = projection[0]
    assert row[shuffled.index("delt_ant_r")] == pytest.approx(0.7)
    assert row[shuffled.index("delt_med_r")] == pytest.approx(0.3)


def test_unknown_actuator_name_is_refused():
    mapping = _mapping()
    mapping["channels"][1]["simulation_actuators"] = ["not_in_model"]

    with pytest.raises(ValueError, match=r"absent from the ordered action vector"):
        build_emg_observation_projection(mapping, ACTUATORS)


def test_weights_that_do_not_sum_to_one_are_refused():
    mapping = _mapping()
    mapping["channels"][0]["weights"] = [0.7, 0.7]

    with pytest.raises(ValueError, match=r"must sum to one"):
        build_emg_observation_projection(mapping, ACTUATORS)


def test_duplicate_actuator_names_are_refused():
    with pytest.raises(ValueError, match=r"must be unique"):
        build_emg_observation_projection(_mapping(), ("a", "b", "a"))


def test_mapping_with_no_comparable_channel_is_refused():
    mapping = _mapping(exclude_last=False)
    for entry in mapping["channels"]:
        entry["mapping_status"] = "excluded_no_verified_model_homolog"

    with pytest.raises(ValueError, match=r"no comparable channel"):
        build_emg_observation_projection(mapping, ACTUATORS)


def test_project_ordered_activation_matches_the_matrix_product():
    spec, _ = _spec()
    activation = jnp.asarray(np.linspace(0.1, 0.9, len(ACTUATORS)), dtype=jnp.float32)

    projected = project_ordered_activation(activation, spec)

    expected = np.asarray(spec.projection) @ np.asarray(activation)
    np.testing.assert_allclose(np.asarray(projected), expected, atol=1e-6)


# ---------------------------------------------------------------- tube shape


def test_distance_is_exactly_zero_inside_the_tube():
    scale = jnp.asarray([0.1, 0.1, 0.1])
    mean = jnp.asarray([0.5, 0.5, 0.5])
    inside = mean + 0.9 * DEFAULT_TUBE_KAPPA * scale

    distance = tube_distance(inside, mean, scale, jnp.ones(3))

    assert float(jnp.max(distance)) == 0.0


def test_distance_grows_once_outside_the_tube():
    scale = jnp.asarray([0.1])
    mean = jnp.asarray([0.5])
    near = mean + (DEFAULT_TUBE_KAPPA + 0.5) * scale
    far = mean + (DEFAULT_TUBE_KAPPA + 2.0) * scale

    assert float(tube_distance(far, mean, scale, jnp.ones(1))[0]) > float(
        tube_distance(near, mean, scale, jnp.ones(1))[0]
    )


def test_distance_is_symmetric_about_the_human_centre():
    scale, mean, valid = jnp.asarray([0.1]), jnp.asarray([0.5]), jnp.ones(1)
    offset = (DEFAULT_TUBE_KAPPA + 1.0) * 0.1

    above = float(tube_distance(mean + offset, mean, scale, valid)[0])
    below = float(tube_distance(mean - offset, mean, scale, valid)[0])

    assert above == pytest.approx(below)


def test_far_deviation_is_charged_linearly_not_quadratically():
    scale, mean, valid = jnp.asarray([0.1]), jnp.asarray([0.0]), jnp.ones(1)
    # Beyond huber_delta the penalty must become linear, so one broken channel
    # cannot dominate the whole term.
    excess = DEFAULT_TUBE_KAPPA + DEFAULT_HUBER_DELTA
    single = float(tube_distance((excess + 10.0) * scale, mean, scale, valid)[0])
    double = float(tube_distance((excess + 20.0) * scale, mean, scale, valid)[0])

    assert (double - single) == pytest.approx(10.0 * DEFAULT_HUBER_DELTA, rel=1e-4)


def test_invalid_bins_contribute_nothing():
    scale, mean = jnp.asarray([0.1, 0.1]), jnp.asarray([0.5, 0.5])
    outside = jnp.asarray([5.0, 5.0])

    distance = tube_distance(outside, mean, scale, jnp.asarray([1.0, 0.0]))

    assert float(distance[1]) == 0.0
    assert float(distance[0]) > 0.0


def test_zero_scale_cannot_produce_a_divide_by_zero():
    distance = tube_distance(jnp.asarray([1.0]), jnp.asarray([0.0]), jnp.asarray([0.0]), jnp.ones(1))

    assert np.isfinite(float(distance[0]))


# ---------------------------------------------------------------- spec build


def test_spec_identity_pins_reference_mapping_and_schema():
    spec, identity = _spec()
    tube = _tube()

    assert spec.signal == EMG_ANCHOR_SIGNAL
    assert spec.channel_names == CHANNELS
    assert spec.channel_count == len(CHANNELS)
    assert spec.synergy_count == 2
    assert identity.reference_fingerprint == tube.reference_fingerprint
    assert identity.mapping_id == "test_mapping_v1"
    assert identity.muscle_channel_core_fingerprint == "d" * 64
    assert len(identity.loss_spec_fingerprint) == 64


def test_spec_fingerprint_is_stable_across_rebuilds():
    first = _spec()[1].loss_spec_fingerprint
    second = _spec()[1].loss_spec_fingerprint

    assert first == second


def test_spec_fingerprint_tracks_the_tube_shape_parameters():
    baseline = _spec()[1].loss_spec_fingerprint
    widened = _spec(tube_kappa=DEFAULT_TUBE_KAPPA + 1.0)[1].loss_spec_fingerprint

    # A different tube width is a different loss; the manifest has to say so.
    assert widened != baseline


def test_channel_order_divergence_between_mapping_and_tube_is_refused():
    mapping = _mapping()
    mapping["channels"][0], mapping["channels"][1] = (
        mapping["channels"][1],
        mapping["channels"][0],
    )

    with pytest.raises(ValueError, match=r"diverges from the reference tube"):
        build_emg_anchor_spec(
            _tube(),
            mapping,
            actuator_names=ACTUATORS,
            activation_addresses=np.arange(len(ACTUATORS), dtype=np.int32),
            muscle_channel_core_fingerprint="d" * 64,
        )


def test_identity_manifest_is_json_serialisable():
    import json

    _, identity = _spec()

    payload = json.loads(json.dumps(identity.to_manifest()))

    assert payload["channel_names"] == list(CHANNELS)
    assert payload["loss_spec_fingerprint"] == identity.loss_spec_fingerprint


# ---------------------------------------------------------------- phase bins


def test_phase_maps_onto_static_bins():
    spec, _ = _spec()

    assert int(phase_bin_index(0.0, spec)) == 0
    assert int(phase_bin_index(0.99, spec)) == PHASE_BINS - 1


def test_out_of_range_phase_is_clipped_not_wrapped():
    spec, _ = _spec()

    assert int(phase_bin_index(-0.5, spec)) == 0
    assert int(phase_bin_index(1.5, spec)) == PHASE_BINS - 1


# ---------------------------------------------------------------- anchor loss


def test_activation_at_the_human_centre_costs_nothing():
    spec, _ = _spec()

    metrics = emg_anchor_metrics(_activation(0.5), spec, action_index=0, phase=0.5)

    assert float(metrics.loss) == pytest.approx(0.0, abs=1e-6)
    assert float(metrics.violation_fraction) == pytest.approx(0.0)
    assert float(metrics.valid_channel_fraction) == pytest.approx(1.0)


def test_saturated_activation_is_penalised_and_flagged():
    spec, _ = _spec()

    metrics = emg_anchor_metrics(_activation(1.0), spec, action_index=0, phase=0.5)

    assert float(metrics.loss) > 0.0
    assert float(metrics.violation_fraction) == pytest.approx(1.0)
    assert float(metrics.max_abs_deviation) > float(metrics.mean_abs_deviation) - 1e-6


def test_unreliable_mvc_reference_downweights_amplitude_without_deleting_channel():
    spec, _ = _spec()
    activation = _activation(0.5).at[ACTUATORS.index("bic_l_r")].set(1.0)
    uniform = spec.replace(
        amplitude_confidence=jnp.ones_like(spec.amplitude_confidence)
    )
    downweighted = spec.replace(
        amplitude_confidence=spec.amplitude_confidence.at[0, 1].set(0.2)
    )

    uniform_metrics = emg_anchor_metrics(
        activation, uniform, action_index=0, phase=0.5
    )
    downweighted_metrics = emg_anchor_metrics(
        activation, downweighted, action_index=0, phase=0.5
    )

    assert float(uniform_metrics.channel_loss[1]) > 0.0
    assert float(downweighted_metrics.channel_loss[1]) == pytest.approx(
        float(uniform_metrics.channel_loss[1])
    )
    assert float(downweighted_metrics.loss) < float(uniform_metrics.loss)
    assert float(downweighted_metrics.valid_channel_fraction) == pytest.approx(1.0)


def test_anchor_diagnostics_expose_per_channel_attribution():
    spec, _ = _spec()

    metrics = emg_anchor_metrics(_activation(1.0), spec, action_index=0, phase=0.5)

    assert metrics.channel_loss.shape == (len(CHANNELS),)
    assert metrics.projected_activation.shape == (len(CHANNELS),)
    assert np.all(np.asarray(metrics.channel_loss) >= 0.0)


def test_anchor_diagnostics_report_masked_measured_channel_correlation():
    spec, _ = _spec()
    activation = jnp.linspace(0.1, 0.9, len(ACTUATORS), dtype=jnp.float32)
    projected = emg_anchor_metrics(
        activation,
        spec,
        action_index=0,
        phase=0.5,
    ).projected_activation
    reference = jnp.broadcast_to(projected, spec.anchor_mean.shape)
    correlated_spec = spec.replace(anchor_mean=reference)

    metrics = emg_anchor_metrics(
        activation,
        correlated_spec,
        action_index=0,
        phase=0.5,
    )

    assert float(metrics.pattern_correlation) == pytest.approx(1.0, abs=1e-5)


def test_loss_never_reads_actuators_outside_the_measured_subspace():
    spec, _ = _spec()
    baseline = _activation(0.5)
    perturbed = baseline.at[ACTUATORS.index("soleus_r")].set(1.0)

    at_centre = float(emg_anchor_metrics(baseline, spec, action_index=0, phase=0.5).loss)
    perturbed_loss = float(emg_anchor_metrics(perturbed, spec, action_index=0, phase=0.5).loss)

    # soleus has no verified electrode homolog, so moving it must be invisible
    # to the anchor term.  This is the claim that the loss respects what was
    # actually measured.
    assert perturbed_loss == pytest.approx(at_centre, abs=1e-6)


def test_anchor_loss_is_differentiable_and_routes_gradient_to_mapped_actuators():
    spec, _ = _spec()

    grad = jax.grad(lambda a: emg_anchor_metrics(a, spec, action_index=0, phase=0.5).loss)(_activation(1.0))

    assert np.all(np.isfinite(np.asarray(grad)))
    assert abs(float(grad[ACTUATORS.index("delt_ant_r")])) > 0.0
    assert float(grad[ACTUATORS.index("soleus_r")]) == pytest.approx(0.0)


def test_anchor_loss_survives_jit_and_vmap():
    spec, _ = _spec()
    batch = jnp.stack([_activation(0.5), _activation(1.0)])

    batched = jax.jit(jax.vmap(lambda a: emg_anchor_metrics(a, spec, action_index=0, phase=0.5).loss))(batch)

    assert batched.shape == (2,)
    assert float(batched[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(batched[1]) > 0.0


def test_invalid_reference_bins_disarm_the_anchor_term():
    # Under-sampled bins are marked invalid, and an all-invalid tube must yield
    # a zero loss rather than a spurious gradient.
    spec, _ = build_emg_anchor_spec(
        _tube(action_envelopes={"forehand_clear": _envelopes(num_trials=1)}),
        _mapping(),
        actuator_names=ACTUATORS,
        activation_addresses=np.arange(len(ACTUATORS), dtype=np.int32),
        muscle_channel_core_fingerprint="d" * 64,
    )

    metrics = emg_anchor_metrics(_activation(1.0), spec, action_index=0, phase=0.5)

    assert float(metrics.loss) == pytest.approx(0.0)
    assert float(metrics.valid_channel_fraction) == pytest.approx(0.0)


# --------------------------------------------------------------- synergy loss


def test_synergy_metrics_split_shape_from_intensity():
    spec, _ = _spec()

    metrics = emg_synergy_metrics(_activation(0.5), spec, action_index=0, phase=0.5)

    assert metrics.coefficients.shape == (2,)
    assert np.all(np.asarray(metrics.coefficients) >= 0.0)
    assert float(metrics.shape_cosine) <= 1.0 + 1e-6
    assert float(metrics.loss) == pytest.approx(float(metrics.shape_loss) + float(metrics.intensity_loss), rel=1e-5)


def test_matching_coordination_scores_high_cosine():
    spec, _ = _spec()

    metrics = emg_synergy_metrics(_activation(0.5), spec, action_index=0, phase=0.5)

    assert float(metrics.shape_cosine) > 0.9


def test_scaling_activation_moves_intensity_but_preserves_shape():
    spec, _ = _spec()

    weak = emg_synergy_metrics(_activation(0.2), spec, action_index=0, phase=0.5)
    strong = emg_synergy_metrics(_activation(0.9), spec, action_index=0, phase=0.5)

    # A uniform gain is an intensity change, not a coordination change: cosine
    # should barely move while intensity clearly does.
    assert float(strong.intensity) > float(weak.intensity)
    assert float(strong.shape_cosine) == pytest.approx(float(weak.shape_cosine), abs=5e-2)


def test_zero_weights_disable_the_synergy_term():
    spec, _ = _spec()

    metrics = emg_synergy_metrics(
        _activation(1.0),
        spec,
        action_index=0,
        phase=0.5,
        shape_weight=0.0,
        intensity_weight=0.0,
    )

    assert float(metrics.loss) == pytest.approx(0.0)


def test_synergy_loss_is_differentiable_under_jit():
    spec, _ = _spec()

    grad = jax.jit(jax.grad(lambda a: emg_synergy_metrics(a, spec, action_index=0, phase=0.5).loss))(_activation(0.9))

    assert np.all(np.isfinite(np.asarray(grad)))
