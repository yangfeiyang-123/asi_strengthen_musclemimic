"""Contract tests for the phase-indexed EMG reference tube.

The tube is the single artifact every EMG-aware training stage reads, so its
schema, its fail-closed reward gate, and its fingerprint stability are all
load-bearing.  Array layout is ``[action, phase_bin, channel]``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from musclemimic.physiology.emg_reference import (
    EMG_REFERENCE_TUBE_SCHEMA_VERSION,
    EMG_SYNERGY_PROJECTION_METHOD,
    EMG_SYNERGY_RIDGE,
    EMG_TUBE_MIN_TRIALS,
    EMG_TUBE_SCALE_FLOOR,
    EMG_TUBE_STATISTIC,
    build_emg_dual_track_normalization,
    build_phase_reference_tube,
    emg_reference_fingerprint,
    load_emg_phase_reference_tube,
    resolve_emg_reference_reward_gate,
    save_emg_phase_reference_tube,
    synergy_projection_matrix,
    validate_emg_phase_reference_tube,
)

CHANNELS = ("deltoid_anterior", "biceps_brachii", "triceps_lateral", "flexor_carpi")
PHASE_BINS = 5


def _envelopes(num_trials=6, num_samples=40, *, seed=0):
    """Per-trial normalised envelopes shaped [trial, sample, channel]."""

    rng = np.random.default_rng(seed)
    phase = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    base = np.stack(
        [
            0.5 + 0.3 * np.sin(2 * np.pi * phase),
            0.4 + 0.2 * np.cos(2 * np.pi * phase),
            0.3 + 0.1 * phase,
            0.2 + 0.1 * np.sin(4 * np.pi * phase),
        ],
        axis=1,
    )
    noisy = base[None, ...] + 0.02 * rng.standard_normal((num_trials, num_samples, len(CHANNELS)))
    return np.clip(noisy, 0.0, 1.0)


def _basis(synergy_count=2):
    columns = [
        np.array([1.0, 0.8, 0.1, 0.0]),
        np.array([0.0, 0.1, 0.9, 0.7]),
        np.array([0.2, 0.0, 0.3, 1.0]),
    ][:synergy_count]
    return np.stack(columns, axis=1)


def _mapping_binding(review_status="verified"):
    return {
        "mapping_id": "p002_forehand_clear_v1",
        "mapping_sha256": "a" * 64,
        "mapping_review_status": review_status,
        "acquired_channel_count": len(CHANNELS),
        "comparable_channel_count": len(CHANNELS),
        "actuator_schema_hash": "b" * 64,
    }


def _synergy_binding(synergy_count=2):
    return {
        "basis_id": "p002_nmf_k2",
        "basis_sha256": "c" * 64,
        "synergy_count": synergy_count,
        "channel_normalization": "unit_variance_per_channel",
        "projection_method": EMG_SYNERGY_PROJECTION_METHOD,
        "projection_ridge": EMG_SYNERGY_RIDGE,
    }


def _build(**overrides):
    payload = {
        "reference_id": "P002_forehand_clear",
        "action_envelopes": {"forehand_clear": _envelopes()},
        "channel_names": CHANNELS,
        "synergy_basis": _basis(),
        "mapping_binding": _mapping_binding(),
        "synergy_binding": _synergy_binding(),
        "provenance": {
            "subject": "P002",
            "session": "2025-07-11",
            "normalization": "mvc_percent",
            "review_evidence": [],
        },
        "phase_bin_count": PHASE_BINS,
    }
    payload.update(overrides)
    if "normalization_binding" not in payload:
        action_samples = {
            action: [values[index] for index in range(values.shape[0])]
            for action, values in payload["action_envelopes"].items()
        }
        payload["normalization_binding"] = build_emg_dual_track_normalization(
            action_samples=action_samples,
            channel_names=payload["channel_names"],
            training_cohorts={
                action: [
                    {
                        "trial_id": f"trial_{index:03d}",
                        "mvc_normalized_emg_sha256": f"{index + 1:064x}",
                    }
                    for index in range(values.shape[0])
                ]
                for action, values in payload["action_envelopes"].items()
            },
            mvc_final_reference_mv=np.ones(len(payload["channel_names"])),
            mvc_reference_binding={
                "path": "/controlled/preprocessing_mvc_reference.json",
                "sha256": "f" * 64,
                "scope": "participant",
                "algorithm": "controlled fixture MVC",
            },
        )
    return build_phase_reference_tube(**payload)


def _trial_qc_review_binding(*, num_trials=6, review_sha256="d" * 64):
    return {
        "schema_version": "emg_trial_channel_qc_review_v1",
        "action": "forehand_clear",
        "review_status": "verified",
        "training_enabled": True,
        "source_path": "/controlled/evidence/emg_trial_qc_review.json",
        "review_sha256": review_sha256,
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
    }


def _reviewed_provenance(*, num_trials=6, review_sha256="d" * 64):
    return {
        "subject": "P002",
        "session": "2025-07-11",
        "normalization": "mvc_percent",
        "review_evidence": [
            {
                "reviewer": "domain_expert",
                "date": "2025-07-20",
                "artifact": "doc/emg_mapping_review.md",
            }
        ],
        "trial_qc_review": _trial_qc_review_binding(
            num_trials=num_trials,
            review_sha256=review_sha256,
        ),
    }


def _verified(**overrides):
    """A tube cleared for training: verified review plus training_enabled."""

    payload = {
        "review_status": "verified",
        "training_enabled": True,
        "provenance": _reviewed_provenance(),
    }
    payload.update(overrides)
    return _build(**payload)


def test_built_tube_carries_schema_statistic_and_scale_floor():
    tube = _build()

    assert tube.schema_version == EMG_REFERENCE_TUBE_SCHEMA_VERSION
    assert tube.statistic == EMG_TUBE_STATISTIC
    assert tube.anchor_mean.shape == (1, PHASE_BINS, len(CHANNELS))
    assert tube.synergy_mean.shape == (1, PHASE_BINS, 2)
    # A robust scale that collapses to zero would turn the tube into a hard
    # equality constraint, so the floor has to survive into the artifact.
    assert np.all(tube.anchor_scale >= EMG_TUBE_SCALE_FLOOR)
    assert np.all(tube.anchor_mean >= 0.0)
    assert np.all(tube.mvc_anchor_mean >= 0.0)
    assert np.all(tube.robust_scale > 0.0)
    assert np.all(tube.synergy_mean >= 0.0)
    assert np.all(tube.anchor_trial_count == 6)
    assert np.all(tube.anchor_valid)


def test_new_tube_defaults_to_diagnostics_only():
    tube = _build()

    # The safe default matters: an unreviewed reference must not silently
    # acquire authority over training.
    assert tube.review_status == "provisional"
    assert tube.training_enabled is False


def test_tube_round_trips_through_disk_unchanged(tmp_path):
    tube = _build()
    save_emg_phase_reference_tube(tube, tmp_path / "ref")
    reloaded = load_emg_phase_reference_tube(tmp_path / "ref")

    np.testing.assert_allclose(reloaded.anchor_mean, tube.anchor_mean)
    np.testing.assert_allclose(reloaded.anchor_scale, tube.anchor_scale)
    np.testing.assert_allclose(reloaded.mvc_anchor_mean, tube.mvc_anchor_mean)
    np.testing.assert_allclose(reloaded.robust_scale, tube.robust_scale)
    np.testing.assert_allclose(
        reloaded.amplitude_confidence, tube.amplitude_confidence
    )
    np.testing.assert_allclose(reloaded.synergy_mean, tube.synergy_mean)
    np.testing.assert_allclose(reloaded.synergy_basis, tube.synergy_basis)
    assert reloaded.channel_names == tube.channel_names
    assert reloaded.action_ids == tube.action_ids
    assert reloaded.reference_fingerprint == tube.reference_fingerprint


def test_fingerprint_is_content_addressed_not_key_order_dependent(tmp_path):
    tube = _build()
    save_emg_phase_reference_tube(tube, tmp_path / "ref")
    manifest_path = tmp_path / "ref" / "emg_reference_manifest.json"
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("reference_fingerprint", None)
    shuffled = dict(reversed(list(manifest.items())))

    assert emg_reference_fingerprint(shuffled) == emg_reference_fingerprint(manifest)


def test_fingerprint_separates_different_references():
    baseline = _build().reference_fingerprint
    renamed = _build(reference_id="P003_forehand_clear").reference_fingerprint

    assert renamed != baseline


def test_too_few_trials_marks_bins_invalid_rather_than_averaging():
    tube = _build(action_envelopes={"forehand_clear": _envelopes(num_trials=EMG_TUBE_MIN_TRIALS - 1)})

    # Under-sampled bins stay present but must not be treated as usable.
    assert not bool(np.any(tube.anchor_valid))
    assert not bool(np.any(tube.synergy_valid))


def test_fewer_samples_than_phase_bins_is_refused():
    with pytest.raises(ValueError, match=r"fewer than \d+ phase bins"):
        _build(
            action_envelopes={"forehand_clear": _envelopes(num_samples=3)},
            phase_bin_count=PHASE_BINS,
        )


def test_super_mvc_is_preserved_in_audit_and_robustly_scaled_without_clipping():
    envelopes = np.full((6, 40, len(CHANNELS)), 1.2, dtype=np.float64)

    tube = _build(action_envelopes={"forehand_clear": envelopes})

    np.testing.assert_allclose(tube.mvc_anchor_mean, 1.2)
    assert float(np.max(tube.mvc_anchor_mean)) > 1.0
    np.testing.assert_allclose(tube.robust_scale, 1.2)
    np.testing.assert_allclose(tube.anchor_mean, 1.0, atol=1e-7)
    assert tube.normalization_binding["actions"][0]["channels"][0][
        "mvc_quality"
    ] == "good"


@pytest.mark.parametrize(
    ("ratio", "quality", "confidence"),
    [
        (1.20, "good", 1.0),
        (1.35, "questionable", 0.7),
        (1.75, "unreliable", 0.4),
        (2.25, "invalid_for_absolute_amplitude", 0.2),
    ],
)
def test_mvc_quality_grades_downweight_without_deleting_channels(
    ratio,
    quality,
    confidence,
):
    envelopes = np.full((6, 40, len(CHANNELS)), ratio, dtype=np.float64)

    tube = _build(action_envelopes={"forehand_clear": envelopes})

    assert np.all(tube.anchor_valid)
    np.testing.assert_allclose(tube.mvc_anchor_mean, ratio)
    np.testing.assert_allclose(tube.amplitude_confidence, confidence)
    assert {
        stats["mvc_quality"]
        for stats in tube.normalization_binding["actions"][0]["channels"]
    } == {quality}


def test_nonfinite_or_negative_task_signal_remains_a_hard_failure():
    nonfinite = np.full((6, 40, len(CHANNELS)), 0.5, dtype=np.float64)
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match=r"finite and non-negative"):
        _build(action_envelopes={"forehand_clear": nonfinite})

    negative = np.full((6, 40, len(CHANNELS)), 0.5, dtype=np.float64)
    negative[0, 0, 0] = -0.1
    with pytest.raises(ValueError, match=r"finite and non-negative"):
        _build(action_envelopes={"forehand_clear": negative})


def test_channel_count_mismatch_between_basis_and_names_is_refused():
    with pytest.raises(ValueError, match=r"synergy_basis must be"):
        _build(synergy_basis=_basis()[:2, :])


def test_empty_action_set_is_refused():
    with pytest.raises(ValueError, match=r"at least one action"):
        _build(action_envelopes={})


def test_provisional_reference_cannot_shape_training():
    tube = _build()

    with pytest.raises(ValueError, match=r"mapping review must complete"):
        resolve_emg_reference_reward_gate(tube, enabled=True)


def test_provisional_channel_mapping_cannot_anchor_a_reward():
    tube = _verified(mapping_binding=_mapping_binding(review_status="provisional"))

    with pytest.raises(ValueError, match=r"provisional mapping cannot anchor"):
        resolve_emg_reference_reward_gate(tube, enabled=True)


def test_verified_reference_requires_recorded_review_evidence():
    # "verified" has to mean a human signed off somewhere citable, otherwise
    # the flag is self-asserted and the gate protects nothing.
    with pytest.raises(ValueError, match=r"non-empty review_evidence"):
        _build(review_status="verified", training_enabled=True)


def test_training_enabled_reference_requires_trial_qc_review_binding():
    provenance = _reviewed_provenance()
    provenance.pop("trial_qc_review")

    with pytest.raises(ValueError, match=r"requires provenance\.trial_qc_review"):
        _build(
            review_status="verified",
            training_enabled=True,
            provenance=provenance,
        )


def test_training_enabled_reference_binds_review_bundle_bytes(tmp_path):
    review_bytes = b'{"controlled":"human-review-fixture"}\n'
    review_sha256 = hashlib.sha256(review_bytes).hexdigest()
    tube = _verified(
        provenance=_reviewed_provenance(review_sha256=review_sha256)
    )
    root = tmp_path / "ref"
    save_emg_phase_reference_tube(tube, root)

    with pytest.raises(FileNotFoundError, match=r"trial-QC review bundle is absent"):
        load_emg_phase_reference_tube(root)

    (root / "emg_trial_qc_review.json").write_bytes(review_bytes)
    loaded = load_emg_phase_reference_tube(root)
    assert loaded.training_enabled is True

    (root / "emg_trial_qc_review.json").write_bytes(review_bytes + b"tampered")
    with pytest.raises(ValueError, match=r"trial-QC review bundle content hash mismatch"):
        load_emg_phase_reference_tube(root)


def test_diagnostics_only_reference_cannot_shape_training():
    tube = _build(
        review_status="verified",
        training_enabled=False,
        provenance=_reviewed_provenance(),
    )

    with pytest.raises(ValueError, match=r"diagnostics only"):
        resolve_emg_reference_reward_gate(tube, enabled=True)


def test_reference_with_no_valid_bin_cannot_shape_training():
    trial_count = EMG_TUBE_MIN_TRIALS - 1
    tube = _verified(
        action_envelopes={"forehand_clear": _envelopes(num_trials=trial_count)},
        provenance=_reviewed_provenance(num_trials=trial_count),
    )

    with pytest.raises(ValueError, match=r"no valid phase bin"):
        resolve_emg_reference_reward_gate(tube, enabled=True)


def test_fully_verified_reference_arms_the_reward():
    active, reason = resolve_emg_reference_reward_gate(_verified(), enabled=True)

    assert active is True
    assert "armed" in reason


def test_disabled_configuration_short_circuits_before_validation():
    # Config-off must be an ordinary no-op, not an error, so ablations that
    # drop the EMG term stay runnable against the same artifact.
    active, reason = resolve_emg_reference_reward_gate(_build(), enabled=False)

    assert active is False
    assert "disabled" in reason


def test_validate_rejects_unknown_schema_version(tmp_path):
    tube = _build()
    save_emg_phase_reference_tube(tube, tmp_path / "ref")
    import json

    manifest_path = tmp_path / "ref" / "emg_reference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "emg_phase_reference_tube_v0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unsupported EMG reference schema"):
        load_emg_phase_reference_tube(tmp_path / "ref")


def test_validate_rejects_incomplete_artifact(tmp_path):
    tube = _build()
    save_emg_phase_reference_tube(tube, tmp_path / "ref")
    (tmp_path / "ref" / "emg_reference_tube.npz").unlink()

    with pytest.raises(FileNotFoundError, match=r"incomplete"):
        load_emg_phase_reference_tube(tmp_path / "ref")


def test_validate_rejects_missing_mapping_binding_field():
    binding = _mapping_binding()
    del binding["actuator_schema_hash"]

    with pytest.raises(ValueError, match=r"actuator_schema_hash"):
        _build(mapping_binding=binding)


def test_synergy_projection_is_nonnegative_and_matches_basis_shape():
    basis = _basis(synergy_count=3)
    projector = synergy_projection_matrix(basis)

    assert projector.shape == (3, len(CHANNELS))
    # Projecting the basis itself should recover a dominant diagonal: each
    # synergy explains its own column better than the others do.
    coefficients = np.maximum(basis.T @ projector.T, 0.0)
    assert np.all(np.diag(coefficients) > 0.0)
    assert np.argmax(coefficients, axis=1).tolist() == [0, 1, 2]


def test_validate_round_trips_a_hand_built_payload(tmp_path):
    tube = _build()
    arrays = {
        name: getattr(tube, name)
        for name in (
            "anchor_mean",
            "anchor_scale",
            "mvc_anchor_mean",
            "mvc_anchor_scale",
            "robust_scale",
            "amplitude_confidence",
            "anchor_valid",
            "anchor_trial_count",
            "synergy_mean",
            "synergy_scale",
            "synergy_valid",
            "synergy_basis",
        )
    }
    identity = {
        "schema_version": tube.schema_version,
        "reference_id": tube.reference_id,
        "review_status": tube.review_status,
        "training_enabled": tube.training_enabled,
        "default_behavior": tube.default_behavior,
        "statistic": tube.statistic,
        "scale_floor": tube.scale_floor,
        "min_trials": tube.min_trials,
        "mapping_binding": dict(tube.mapping_binding),
        "synergy_binding": dict(tube.synergy_binding),
        "normalization_binding": dict(tube.normalization_binding),
        "action_ids": list(tube.action_ids),
        "channel_names": list(tube.channel_names),
        "phase_bin_count": tube.phase_bin_count,
        "provenance": dict(tube.provenance),
    }

    rebuilt = validate_emg_phase_reference_tube(identity, arrays)

    assert rebuilt.reference_fingerprint == tube.reference_fingerprint
