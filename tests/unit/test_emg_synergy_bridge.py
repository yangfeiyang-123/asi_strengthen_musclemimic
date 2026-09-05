"""Bridge from a measured Jidian synergy basis to the 15-channel comparison space."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from musclemimic.evaluation.emg_synergy_bridge import (
    comparable_channel_names,
    compare_measured_and_simulated_synergies,
    load_measured_synergy_basis,
    project_activation_to_comparable_channels,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = (
    REPOSITORY_ROOT / "configs" / "physiology" / "emg_badminton_synergy_16_v2_myofullbody_observation_v1.json"
)
TAXONOMY_PATH = REPOSITORY_ROOT / "configs" / "physiology" / "myofullbody_354_muscle_taxonomy_audit_v2.json"

SENSORS: tuple[tuple[int, str, str], ...] = (
    (1, "right", "upper_trapezius"),
    (2, "right", "anterior_deltoid"),
    (3, "right", "posterior_deltoid"),
    (4, "right", "pectoralis_major_clavicular"),
    (5, "right", "latissimus_dorsi"),
    (6, "right", "triceps_lateral"),
    (7, "right", "pronator_teres"),
    (8, "right", "extensor_carpi_radialis"),
    (9, "right", "external_oblique"),
    (10, "left", "external_oblique"),
    (11, "right", "vastus_lateralis"),
    (12, "left", "vastus_lateralis"),
    (13, "right", "biceps_femoris_long_head"),
    (14, "left", "biceps_femoris_long_head"),
    (15, "right", "gastrocnemius_medialis"),
    (16, "left", "gastrocnemius_medialis"),
)


def _mapping() -> dict:
    return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))


def _actuator_names() -> list[str]:
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    return [row["name"] for row in taxonomy["ordered_actuators"]]


def _write_measured_artifact(
    directory: Path,
    basis: np.ndarray,
    *,
    channel_normalization: str = "unit_variance",
    profile_id: str = "badminton_synergy_16_v2",
    profile_version: int = 2,
    sensor_ids: tuple[int, ...] | None = None,
    selection_is_fallback: bool | None = None,
    selected_k_is_reproducible: bool | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    ids = sensor_ids if sensor_ids is not None else tuple(item[0] for item in SENSORS)
    np.savez(
        directory / "synergy_basis.npz",
        W=basis.astype(np.float32),
        W_recorded_units=basis.astype(np.float32),
        channel_scale=np.ones(basis.shape[0]),
        channel_ids=np.asarray(ids, dtype=np.int16),
        muscle_slugs=np.asarray([item[2] for item in SENSORS], dtype="<U27"),
        sides=np.asarray([item[1] for item in SENSORS], dtype="<U5"),
        k=np.asarray(basis.shape[1], dtype=np.int16),
        fs_hz=np.asarray(2000.0),
    )
    (directory / "synergy_metadata.json").write_text(
        json.dumps(
            {
                "profile_id": profile_id,
                "profile_version": profile_version,
                "global_vaf": 0.88,
                "recorded_unit_global_vaf": 0.81,
                "dataset_sha256": "a" * 64,
                "actions": ["forehand_high_clear"],
                "participants": ["P002"],
                "selection_is_fallback_largest_k": selection_is_fallback,
                "selected_k_is_reproducible": selected_k_is_reproducible,
                "nmf_parameters": {"channel_normalization": channel_normalization},
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_measured_basis_drops_only_the_electrode_the_mapping_excludes(tmp_path):
    rng = np.random.default_rng(0)
    basis = rng.uniform(0.1, 1.0, size=(16, 4))
    artifact = _write_measured_artifact(tmp_path / "fit", basis)

    measured = load_measured_synergy_basis(artifact, mapping=_mapping())

    assert measured["basis"].shape == (15, 4)
    assert measured["channel_names"] == comparable_channel_names(_mapping())
    assert [item["sensor_id"] for item in measured["excluded_channels"]] == [1]
    # Sensor 1 is dropped and every other row survives in the recorded order.
    np.testing.assert_allclose(measured["basis"], basis[1:, :], rtol=1e-6)


def test_measured_basis_refuses_a_reordered_or_foreign_profile(tmp_path):
    basis = np.random.default_rng(1).uniform(0.1, 1.0, size=(16, 3))

    reordered = _write_measured_artifact(
        tmp_path / "reordered", basis, sensor_ids=(2, 1, *range(3, 17))
    )
    with pytest.raises(ValueError, match="channel order"):
        load_measured_synergy_basis(reordered, mapping=_mapping())

    foreign = _write_measured_artifact(tmp_path / "foreign", basis, profile_id="badminton_synergy_16_v1")
    with pytest.raises(ValueError, match="does not match the mapping"):
        load_measured_synergy_basis(foreign, mapping=_mapping())


def test_measured_basis_refuses_a_differently_balanced_fit(tmp_path):
    """The basis depends on which space the fit balanced, so it cannot be rescaled after the fact."""
    basis = np.random.default_rng(2).uniform(0.1, 1.0, size=(16, 3))
    artifact = _write_measured_artifact(tmp_path / "raw", basis, channel_normalization="none")
    with pytest.raises(ValueError, match="refit rather than rescale"):
        load_measured_synergy_basis(artifact, mapping=_mapping(), required_channel_normalization="unit_variance")


def test_simulation_generated_by_a_known_basis_is_recovered_in_the_comparable_space(tmp_path):
    """End-to-end: project actuator activation, refit, and recover the generating structure."""
    mapping = _mapping()
    actuator_names = _actuator_names()
    rng = np.random.default_rng(3)
    rank = 3
    samples = np.arange(400)

    # Build activation whose projection onto the 15 channels is exactly a rank-3
    # product, then check the bridge recovers that structure from the actuators.
    # Each synergy loads mainly on its own group of channels, the way a real one
    # does.  Columns drawn independently over the whole range point in nearly the
    # same direction and NMF cannot separate them from any other spanning set.
    channel_basis = rng.uniform(0.0, 0.08, size=(15, rank))
    for index in range(rank):
        channel_basis[index * 5 : (index + 1) * 5, index] += rng.uniform(0.5, 1.0, size=5)
    coefficients = np.stack(
        [np.exp(-0.5 * ((samples - centre) / 30.0) ** 2) for centre in (80.0, 200.0, 320.0)],
        axis=1,
    )
    channel_signal = coefficients @ channel_basis.T

    contract_channels = [
        entry
        for entry in mapping["channels"]
        if entry.get("mapping_status") != "excluded_no_verified_model_homolog"
    ]
    activation = np.zeros((1, samples.size, len(actuator_names)))
    for index, entry in enumerate(contract_channels):
        # Weights within a channel sum to one, so writing the channel value to
        # each of its actuators makes the projection reproduce it exactly.
        for name in entry["simulation_actuators"]:
            activation[0, :, actuator_names.index(name)] = channel_signal[:, index]

    matrix, channel_names = project_activation_to_comparable_channels(
        activation,
        actuator_names=actuator_names,
        mapping=mapping,
        allow_provisional_mapping=True,
    )
    # The activation contract stores float32, so agreement is exact to that precision.
    np.testing.assert_allclose(matrix, channel_signal, rtol=1e-6, atol=1e-9)

    # The bridge divides the simulated matrix by its own per-channel scale, so
    # `matrix = coefficients @ channel_basis.T` becomes
    # `coefficients @ (channel_basis / scale[:, None]).T`.  The measured artifact
    # stores an already-balanced basis, so it must be divided the same way.
    # Multiplying instead still clears a 0.95 bar here only because the synthetic
    # scale spread is small; on the real data it spans ~70x.
    scale = matrix.std(axis=0)  # must be the statistic the bridge's default uses
    artifact_basis = np.zeros((16, rank))
    artifact_basis[1:, :] = channel_basis / scale[:, None]
    artifact = _write_measured_artifact(tmp_path / "fit", artifact_basis)
    measured = load_measured_synergy_basis(artifact, mapping=mapping)

    report = compare_measured_and_simulated_synergies(
        measured,
        matrix,
        simulated_channel_names=channel_names,
        seeds=(0, 1, 2),
    )
    assert report["rank"] == rank
    assert report["simulated_global_vaf"] > 0.99
    assert report["minimum_weight_cosine_similarity"] > 0.99
    assert report["n_synergies_matched_ge_0_80"] == rank

    # A basis scaled the wrong way must not clear the same bar, or the test above
    # would pass on a bridge that had the convention backwards.
    mis_scaled = np.zeros((16, rank))
    mis_scaled[1:, :] = channel_basis * scale[:, None]
    mis_report = compare_measured_and_simulated_synergies(
        load_measured_synergy_basis(_write_measured_artifact(tmp_path / "bad", mis_scaled), mapping=mapping),
        matrix,
        simulated_channel_names=channel_names,
        seeds=(0, 1, 2),
    )
    assert mis_report["minimum_weight_cosine_similarity"] < report["minimum_weight_cosine_similarity"]


def test_measured_basis_refuses_a_rank_that_was_never_selected(tmp_path):
    """A fallback K sets the rank for both sides, so it must not pass silently."""
    basis = np.random.default_rng(6).uniform(0.1, 1.0, size=(16, 8))
    artifact = _write_measured_artifact(tmp_path / "fallback", basis, selection_is_fallback=True)
    with pytest.raises(ValueError, match="largest tried"):
        load_measured_synergy_basis(artifact, mapping=_mapping())
    accepted = load_measured_synergy_basis(artifact, mapping=_mapping(), allow_fallback_rank=True)
    assert accepted["rank"] == 8


def test_measured_basis_refuses_a_rank_that_does_not_survive_resampling(tmp_path):
    """Reconstruction thresholds can select a K that the fallback flag does not catch."""
    basis = np.random.default_rng(9).uniform(0.1, 1.0, size=(16, 8))
    artifact = _write_measured_artifact(
        tmp_path / "unstable", basis, selection_is_fallback=False, selected_k_is_reproducible=False
    )
    with pytest.raises(ValueError, match="do not reappear when the trials are resampled"):
        load_measured_synergy_basis(artifact, mapping=_mapping())
    assert load_measured_synergy_basis(artifact, mapping=_mapping(), allow_fallback_rank=True)["rank"] == 8


def test_comparison_refuses_two_different_balancing_rules(tmp_path):
    basis = np.random.default_rng(7).uniform(0.1, 1.0, size=(16, 2))
    measured = load_measured_synergy_basis(_write_measured_artifact(tmp_path / "fit", basis), mapping=_mapping())
    matrix = np.abs(np.random.default_rng(8).normal(size=(60, 15))) + 0.1
    with pytest.raises(ValueError, match="same one on both sides"):
        compare_measured_and_simulated_synergies(
            measured,
            matrix,
            simulated_channel_names=measured["channel_names"],
            channel_normalization="unit_max",
        )


def test_comparison_refuses_channel_orders_that_do_not_agree(tmp_path):
    basis = np.random.default_rng(4).uniform(0.1, 1.0, size=(16, 2))
    measured = load_measured_synergy_basis(_write_measured_artifact(tmp_path / "fit", basis), mapping=_mapping())
    matrix = np.abs(np.random.default_rng(5).normal(size=(50, 15))) + 0.1
    with pytest.raises(ValueError, match="channel names differ"):
        compare_measured_and_simulated_synergies(
            measured, matrix, simulated_channel_names=list(reversed(measured["channel_names"]))
        )
