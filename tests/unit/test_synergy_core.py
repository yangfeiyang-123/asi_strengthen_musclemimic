import json

import numpy as np
import pytest

from musclemimic.distill.physical import (
    MUSCLE_ACTIVATION_SOURCE,
    MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
    MUSCLE_EXCITATION_FORMULA,
    MUSCLE_EXCITATION_ROUNDOFF_POLICY,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
    UNIT_INTERVAL_ROUNDOFF_POLICY,
)
from musclemimic.synergy.basis_artifact import load_synergy_basis, save_synergy_basis
from musclemimic.synergy.collect import ctrl_to_unit_excitation
from musclemimic.synergy.grouping import explicit_groups, global_group
from musclemimic.synergy.metrics import global_vaf
from musclemimic.synergy.nmf import fit_best_initialization, transform_nmf
from musclemimic.synergy.preprocess import (
    apply_preprocessor,
    fit_preprocessor,
    phase_balanced_weights,
)
from musclemimic.synergy.schema import (
    ACTIVATION_SIGNAL_KIND,
    EXCITATION_SIGNAL_KIND,
    SignalTransform,
    validate_nmf_signal,
)
from musclemimic.synergy.stability import (
    bootstrap_stability,
    initialization_stability,
    match_bases,
    split_half_stability,
)


def _synthetic_synergy(seed=5, samples=160):
    rng = np.random.default_rng(seed)
    basis = np.asarray(
        [
            [1.0, 0.05],
            [0.8, 0.10],
            [0.05, 1.0],
            [0.10, 0.8],
            [0.4, 0.4],
            [0.9, 0.02],
        ],
        dtype=np.float64,
    )
    coefficients = rng.gamma(shape=2.0, scale=0.4, size=(samples, 2))
    return coefficients @ basis.T, basis


def _activation_transform(names):
    contract = _muscle_contract(names)
    return SignalTransform(
        kind="identity_nonnegative_activation",
        raw_signal_kind=MUSCLE_ACTIVATION_SOURCE,
        formula="activation",
        actuator_names=tuple(names),
        roundoff_policy=UNIT_INTERVAL_ROUNDOFF_POLICY,
        physical_signal_schema_version=PHYSICAL_SIGNAL_SCHEMA_VERSION,
        muscle_channel_contract=contract,
    )


def _muscle_contract(names):
    width = len(names)
    return {
        "schema_version": MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
        "actuator_names": list(names),
        "actuator_ids": list(range(width)),
        "actuator_dyntype": ["muscle"] * width,
        "actuator_actnum": [1] * width,
        "actuator_actadr": list(range(width)),
        "model_na": width,
    }


def test_signed_controls_fail_closed_and_verified_muscle_clip_is_auditable():
    names = ("hip", "shoulder")
    raw = np.asarray([[-0.1, 0.0], [0.5, 1.0], [1.0, 1.2]], dtype=np.float64)
    ctrlrange = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="signed/raw control"):
        validate_nmf_signal(raw, signal_kind="teacher_ctrl_physical", muscle_names=names)

    signal = ctrl_to_unit_excitation(
        raw,
        ctrlrange=ctrlrange,
        actuator_names=names,
        muscle_channel_contract=_muscle_contract(names),
    )

    np.testing.assert_allclose(signal.values, [[0.0, 0.0], [0.5, 1.0], [1.0, 1.0]])
    assert signal.signal_kind == EXCITATION_SIGNAL_KIND
    assert signal.transform is not None
    assert signal.transform.kind == UNIT_EXCITATION_TRANSFORM
    assert signal.transform.formula == MUSCLE_EXCITATION_FORMULA
    assert signal.transform.actuator_names == names
    assert len(signal.transform.ctrlrange_schema_hash) == 64
    assert signal.transform.roundoff_policy == MUSCLE_EXCITATION_ROUNDOFF_POLICY
    assert signal.transform.physical_signal_schema_version == PHYSICAL_SIGNAL_SCHEMA_VERSION
    with pytest.raises(ValueError, match="legacy signed/mixed"):
        ctrl_to_unit_excitation(
            np.asarray([[0.0, 0.5], [0.0, 1.0]]),
            ctrlrange=np.asarray([[-1.0, 1.0], [0.0, 1.0]]),
            actuator_names=names,
            muscle_channel_contract=_muscle_contract(names),
        )


def test_negative_activation_is_never_clipped():
    with pytest.raises(ValueError, match="negative values"):
        validate_nmf_signal(
            np.asarray([[0.1, -0.01], [0.2, 0.3]]),
            signal_kind=ACTIVATION_SIGNAL_KIND,
            muscle_names=("a", "b"),
            transform=_activation_transform(("a", "b")),
        )
    with pytest.raises(ValueError, match=r"must lie in \[0,1\]"):
        validate_nmf_signal(
            np.asarray([[0.1, 1.01], [0.2, 0.3]]),
            signal_kind=ACTIVATION_SIGNAL_KIND,
            muscle_names=("a", "b"),
            transform=_activation_transform(("a", "b")),
        )


def test_preprocessing_is_train_only_and_phase_weights_emphasize_impact():
    names = ("a", "b", "silent")
    transform = _activation_transform(names)
    train = np.asarray([[0.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.25, 0.25, 0.0]])
    val = np.asarray([[1.0, 1.0, 0.0], [0.75, 0.5, 0.0]])
    processed, state = fit_preprocessor(
        train,
        muscle_names=names,
        signal_kind=ACTIVATION_SIGNAL_KIND,
        transform=transform,
        normalization="channel_max",
    )
    transformed_val = apply_preprocessor(
        val,
        state,
        signal_kind=ACTIVATION_SIGNAL_KIND,
        transform=transform,
    )

    np.testing.assert_allclose(state.scales, [0.5, 1.0])
    assert state.kept_muscle_names == ("a", "b")
    np.testing.assert_allclose(processed[1], [1.0, 1.0])
    np.testing.assert_allclose(transformed_val[0], [2.0, 1.0])
    weights = phase_balanced_weights(np.asarray([0, 3, 0, 3], dtype=np.int32))
    assert weights[1] / weights[0] == pytest.approx(4.0)
    assert np.mean(weights) == pytest.approx(1.0)


def test_nmf_reconstructs_heldout_data_and_stability_is_permutation_invariant():
    values, _ = _synthetic_synergy()
    train, val = values[:120], values[120:]
    best, initializations = fit_best_initialization(
        train,
        rank=2,
        seeds=(0, 1, 2),
        max_iter=1200,
        tol=1e-8,
    )
    _, heldout = transform_nmf(val, best.basis)

    assert global_vaf(val, heldout) > 0.995
    assert initialization_stability(initializations)["mean_similarity"] > 0.90
    permuted = best.basis[:, [1, 0]]
    matched = match_bases(best.basis, permuted)
    assert matched["mean_similarity"] == pytest.approx(1.0)
    assert matched["candidate_permutation"] == [1, 0]

    split = split_half_stability(train, rank=2, repeats=2, seed=12, max_iter=500)
    bootstrap = bootstrap_stability(
        train,
        reference_basis=best.basis,
        rank=2,
        repeats=2,
        seed=13,
        max_iter=500,
    )
    assert split["repeats"] == 2
    assert split["mean_similarity"] > 0.75
    assert bootstrap["mean_similarity"] > 0.75


def test_grouping_requires_explicit_unique_complete_ownership():
    names = ("left_leg", "right_leg", "trunk", "right_arm")
    assert global_group(names) == {"whole_body": (0, 1, 2, 3)}
    groups = explicit_groups(
        names,
        {"lower": ["left_leg", "right_leg"], "upper": ["trunk", "right_arm"]},
    )
    assert groups == {"lower": (0, 1), "upper": (2, 3)}
    with pytest.raises(ValueError, match="overlaps"):
        explicit_groups(names, {"a": ["trunk"], "b": ["trunk", "right_arm"]})
    with pytest.raises(ValueError, match="unassigned"):
        explicit_groups(names, {"lower": ["left_leg", "right_leg"]})


def test_basis_artifact_verifies_names_provenance_and_content(tmp_path):
    basis = np.asarray([[0.8, 0.1], [0.1, 0.9], [0.4, 0.4]], dtype=np.float64)
    manifest = {
        "signal_kind": EXCITATION_SIGNAL_KIND,
        "region": "whole_body",
        "rank": 2,
        "normalization": {"normalization": "channel_max", "fit_scope": "train_only"},
        "source_dataset_fingerprint": "dataset-sha256",
        "teacher_checkpoint_fingerprint": "a" * 64,
        "fit_seed": 0,
        "transform": {
            "kind": UNIT_EXCITATION_TRANSFORM,
            "formula": MUSCLE_EXCITATION_FORMULA,
        },
        "split_provenance": {
            "train": {"motion_uids": [1, 2]},
            "validation": {"motion_uids": [3]},
        },
        "train_motion_uids": [1, 2],
    }
    artifact = save_synergy_basis(
        tmp_path / "basis",
        basis=basis,
        muscle_names=("a", "b", "c"),
        manifest=manifest,
    )
    loaded = load_synergy_basis(artifact.path)

    np.testing.assert_allclose(loaded.basis, basis.astype(np.float32))
    assert loaded.muscle_names == ("a", "b", "c")
    assert loaded.manifest["split_provenance"]["validation"]["motion_uids"] == [3]
    assert loaded.fingerprint == artifact.fingerprint
    assert len(loaded.fingerprint) == 64

    manifest_path = artifact.path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["region"] = "tampered"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_fingerprint mismatch"):
        load_synergy_basis(artifact.path)
