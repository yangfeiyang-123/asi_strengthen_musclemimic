"""Cross-action synergy reuse measures."""

from __future__ import annotations

import numpy as np
import pytest

from emg.synergy_reuse import (
    basis_geometry,
    bootstrap_stability,
    heldout_reconstruction,
    initialization_stability,
    project_onto_basis,
    shared_basis_recruitment,
    shared_channel_scale,
    synergy_novelty,
    within_action_heldout,
)


def _bursts(centres, samples=300, width=25.0):
    grid = np.arange(samples)
    return np.stack([np.exp(-0.5 * ((grid - centre) / width) ** 2) for centre in centres])


def test_heldout_vaf_is_near_one_only_when_the_basis_actually_spans_the_target():
    rng = np.random.default_rng(0)
    basis = rng.uniform(0.2, 1.0, size=(16, 3))
    generated = basis @ _bursts((60.0, 150.0, 240.0))
    # A target the basis produced is fully explained; one built from an
    # unrelated basis is not, and the gap is what the reuse claim rests on.
    unrelated = rng.uniform(0.2, 1.0, size=(16, 3)) @ _bursts((80.0, 160.0, 230.0))

    assert heldout_reconstruction(basis, generated)["heldout_global_vaf"] > 0.999
    assert heldout_reconstruction(basis, unrelated)["heldout_global_vaf"] < 0.99


def test_projection_onto_a_fixed_basis_stays_non_negative():
    rng = np.random.default_rng(1)
    basis = rng.uniform(0.2, 1.0, size=(16, 3))
    values = np.abs(rng.normal(size=(16, 50)))
    coefficients, reconstruction = project_onto_basis(basis, values)
    assert coefficients.shape == (3, 50)
    assert (coefficients >= 0.0).all()
    np.testing.assert_allclose(reconstruction, basis @ coefficients, rtol=1e-10)


def test_novelty_separates_a_duplicate_a_combination_and_a_genuinely_new_direction():
    reference = np.zeros((16, 2))
    reference[0:4, 0] = 1.0
    reference[4:8, 1] = 1.0
    candidate = np.column_stack(
        [
            reference[:, 0],  # exact duplicate
            reference[:, 0] + reference[:, 1],  # inside the cone, matches neither column alone
            np.concatenate([np.zeros(8), np.ones(4), np.zeros(4)]),  # outside the cone
        ]
    )

    report = synergy_novelty(reference, candidate)
    flags = [item["is_novel"] for item in report["columns"]]
    assert flags == [False, False, True]
    assert report["novel_synergy_count"] == 1
    # The combination is reachable from the cone even though no single reference
    # column resembles it, which is why the residual and the cosine are both needed.
    combination = report["columns"][1]
    assert combination["cone_residual_ratio"] < 1e-6
    assert combination["max_reference_cosine"] < 0.95


def test_effective_rank_counts_directions_that_carry_energy():
    orthogonal = np.eye(16)[:, :4]
    geometry = basis_geometry(orthogonal)
    assert geometry["effective_rank"] == pytest.approx(4.0, abs=1e-9)
    assert geometry["condition_number"] == pytest.approx(1.0, abs=1e-9)

    # Four columns that nearly coincide span far fewer than four directions.
    collapsed = np.tile(np.linspace(1.0, 2.0, 16)[:, None], (1, 4)) + 1e-3 * np.eye(16)[:, :4]
    assert basis_geometry(collapsed)["effective_rank"] < 1.5


def test_within_action_heldout_is_the_ceiling_a_cross_action_number_is_read_against():
    """Same structure across trials scores high; a different structure does not."""
    rng = np.random.default_rng(2)
    basis = rng.uniform(0.2, 1.0, size=(16, 2))
    trials = [basis @ _bursts((70.0, 200.0), samples=120) + 0.01 * rng.random((16, 120)) for _ in range(6)]
    values = np.concatenate(trials, axis=1)
    boundaries = np.arange(7) * 120

    ceiling = within_action_heldout(values, boundaries, 2, 6, 3, repeats=5)
    assert ceiling["available"]
    assert ceiling["mean_heldout_global_vaf"] > 0.95

    foreign = rng.uniform(0.2, 1.0, size=(16, 2)) @ _bursts((70.0, 200.0), samples=120)
    assert heldout_reconstruction(basis, foreign)["heldout_global_vaf"] < ceiling["mean_heldout_global_vaf"]


def test_stability_helpers_report_agreement_and_refuse_degenerate_requests():
    rng = np.random.default_rng(4)
    basis = rng.uniform(0.2, 1.0, size=(16, 2))
    trials = [basis @ _bursts((70.0, 200.0), samples=120) + 0.01 * rng.random((16, 120)) for _ in range(6)]
    values = np.concatenate(trials, axis=1)
    boundaries = np.arange(7) * 120

    assert initialization_stability(values, 2, 6, 0, restarts=3)["minimum_cosine_similarity"] > 0.9
    fitted = bootstrap_stability(values, np.abs(basis), 2, 6, 0, boundaries, repeats=4)
    assert fitted["available"] and fitted["mean_cosine_similarity"] > 0.9

    with pytest.raises(ValueError, match="repeats >= 1"):
        bootstrap_stability(values, np.abs(basis), 2, 6, 0, boundaries, repeats=0)
    with pytest.raises(ValueError, match="at least 2 restarts"):
        initialization_stability(values, 2, 6, 0, restarts=1)


def test_shared_basis_recruitment_separates_how_much_from_when():
    """Two actions built from the same synergies but used differently."""
    basis = np.zeros((16, 2))
    basis[0:8, 0] = 1.0
    basis[8:16, 1] = 1.0
    early, late = _bursts((40.0,), samples=101, width=12.0), _bursts((85.0,), samples=101, width=12.0)
    # First action is mostly synergy 0 and early; second is mostly synergy 1 and late.
    first = basis @ np.vstack([4.0 * early, 1.0 * late])
    second = basis @ np.vstack([1.0 * early, 4.0 * late])

    report = shared_basis_recruitment(
        basis, {"first": first, "second": second}, scale=np.ones(16), time_normalize_samples=101
    )
    first_block, second_block = report["actions"]["first"], report["actions"]["second"]
    assert first_block["recruitment_share"][0] > second_block["recruitment_share"][0]
    assert second_block["recruitment_share"][1] > first_block["recruitment_share"][1]
    # Peak phase reports when each synergy is recruited, independent of how much.
    assert first_block["peak_phase_percent"][0] == pytest.approx(40.0, abs=1.5)
    assert first_block["peak_phase_percent"][1] == pytest.approx(85.0, abs=1.5)
    assert first_block["heldout_global_vaf"] > 0.999


def test_shared_scale_puts_every_action_in_one_space():
    first = np.abs(np.random.default_rng(5).normal(size=(4, 30))) + 0.1
    second = first * 10.0
    scale = shared_channel_scale({"a": first, "b": second}, "unit_variance")
    # One divisor for both, so a basis fitted on either lands in the same space.
    assert scale.shape == (4,)
    combined = shared_channel_scale({"only": np.concatenate([first, second], axis=1)}, "unit_variance")
    np.testing.assert_allclose(scale, combined, rtol=1e-12)
