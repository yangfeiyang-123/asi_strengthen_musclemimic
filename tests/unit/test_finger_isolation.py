from __future__ import annotations

import numpy as np
import pytest

from musclemimic.utils.finger_isolation import (
    LEFT_FINGER_ACTUATOR_NAMES,
    RIGHT_FINGER_ACTUATOR_NAMES,
    FingerActuatorPartition,
    NamedObservationSchema,
    ObservationField,
    PairedMetricRule,
    SchemaMismatchError,
    compare_paired_metrics,
)


def _full_actuator_names() -> list[str]:
    body = [f"body_{index:03d}" for index in range(354)]
    # Deliberately interleave the three owners.  The implementation must use
    # names and preserve model order instead of relying on positional slices.
    return (
        body[:113]
        + list(RIGHT_FINGER_ACTUATOR_NAMES[:9])
        + body[113:271]
        + list(LEFT_FINGER_ACTUATOR_NAMES[:17])
        + list(RIGHT_FINGER_ACTUATOR_NAMES[9:])
        + body[271:]
        + list(LEFT_FINGER_ACTUATOR_NAMES[17:])
    )


def test_full_partition_is_name_based_and_has_explicit_left_neutral_owner():
    all_names = _full_actuator_names()

    partition = FingerActuatorPartition.from_actuator_names(all_names)

    assert partition.full_size == 416
    assert partition.body_size == 354
    assert partition.right_grip_size == 31
    assert partition.left_neutral_size == 31
    assert set(partition.body_actuator_names).isdisjoint(partition.right_grip_actuator_names)
    assert set(partition.body_actuator_names).isdisjoint(partition.left_neutral_actuator_names)
    assert set(partition.right_grip_actuator_names).isdisjoint(partition.left_neutral_actuator_names)
    assert partition.source_labels().count("body") == 354
    assert partition.source_labels().count("right_grip") == 31
    assert partition.source_labels().count("left_neutral") == 31

    body = np.linspace(-0.5, 0.5, partition.body_size)
    right = np.full(partition.right_grip_size, 0.25)
    left = np.full(partition.left_neutral_size, -0.125)
    merged = partition.merge(body_action=body, right_grip_action=right, left_neutral_action=left)

    np.testing.assert_allclose(merged[partition.body_indices], body)
    np.testing.assert_allclose(merged[partition.right_grip_indices], right)
    np.testing.assert_allclose(merged[partition.left_neutral_indices], left)


def test_partition_rejects_missing_owner_wrong_dimensions_and_name_hash_changes():
    all_names = _full_actuator_names()
    partition = FingerActuatorPartition.from_actuator_names(all_names)

    with pytest.raises(ValueError, match="left_neutral.*31"):
        FingerActuatorPartition.from_actuator_names(all_names[:-1])

    with pytest.raises(ValueError, match="body_action must have shape"):
        partition.merge(
            body_action=np.zeros(353),
            right_grip_action=np.zeros(31),
            left_neutral_action=np.zeros(31),
        )

    changed_order = all_names.copy()
    changed_order[0], changed_order[1] = changed_order[1], changed_order[0]
    with pytest.raises(SchemaMismatchError, match="actuator schema hash mismatch"):
        partition.assert_compatible(changed_order)

    assert partition.schema_hash == FingerActuatorPartition.from_actuator_names(all_names).schema_hash


def test_observation_filter_removes_fingers_by_joint_actuator_and_feature_name():
    schema = NamedObservationSchema(
        fields=(
            ObservationField("root_pose", width=7),
            ObservationField("shoulder_qpos", joint_name="shoulder_elv_r"),
            ObservationField("right_thumb_qpos", joint_name="cmc_flexion_r"),
            ObservationField("left_index_qvel", joint_name="mcp2_flexion_l"),
            ObservationField("right_fds_length", actuator_name="FDS2"),
            ObservationField("left_fds_activation", actuator_name="FDS2_left"),
            ObservationField("right_finger_contact", width=2),
            ObservationField("phase"),
        )
    )
    obs_filter = schema.without_fingers(finger_feature_names={"right_finger_contact"})
    source = np.arange(schema.total_size, dtype=float)

    filtered = obs_filter.apply(source)

    assert obs_filter.removed_feature_names == (
        "right_thumb_qpos",
        "left_index_qvel",
        "right_fds_length",
        "left_fds_activation",
        "right_finger_contact",
    )
    assert obs_filter.target_schema.feature_names == ("root_pose", "shoulder_qpos", "phase")
    np.testing.assert_array_equal(filtered, source[obs_filter.kept_indices])
    batched = np.stack([source, source + 100.0])
    np.testing.assert_array_equal(obs_filter.apply(batched), batched[..., obs_filter.kept_indices])


def test_observation_filter_can_remove_only_right_hand_and_fails_fast_on_schema_drift():
    schema = NamedObservationSchema(
        fields=(
            ObservationField("root"),
            ObservationField("right_joint", joint_name="ip_flexion_r"),
            ObservationField("left_joint", joint_name="ip_flexion_l"),
            ObservationField("right_act", actuator_name="FPL"),
            ObservationField("left_act", actuator_name="FPL_left"),
        )
    )
    obs_filter = schema.without_fingers(sides=("right",))

    assert obs_filter.target_schema.feature_names == ("root", "left_joint", "left_act")
    with pytest.raises(ValueError, match="last dimension"):
        obs_filter.apply(np.zeros(schema.total_size + 1))

    drifted = NamedObservationSchema(fields=schema.fields[::-1])
    with pytest.raises(SchemaMismatchError, match="observation schema hash mismatch"):
        obs_filter.assert_source_schema(drifted)


def test_paired_metrics_report_uses_same_seed_pairs_and_threshold_direction():
    rules = (
        PairedMetricRule("body_site_error", lower_is_better=True, max_relative_degradation=0.05),
        PairedMetricRule("racket_head_error", lower_is_better=True, max_relative_degradation=0.05),
        PairedMetricRule("early_termination", lower_is_better=True, max_absolute_degradation=0.02),
        PairedMetricRule("coverage", lower_is_better=False, max_relative_degradation=0.05),
    )
    clean = {
        "body_site_error": np.array([1.0, 1.0, 1.0]),
        "racket_head_error": np.array([2.0, 2.0, 2.0]),
        "early_termination": np.array([0.0, 0.0, 0.0]),
        "coverage": np.array([0.98, 0.97, 0.96]),
    }
    perturbed = {
        "body_site_error": np.array([1.02, 1.03, 1.01]),
        "racket_head_error": np.array([2.05, 2.04, 2.03]),
        "early_termination": np.array([0.0, 0.0, 0.0]),
        "coverage": np.array([0.97, 0.96, 0.95]),
    }

    report = compare_paired_metrics(
        clean,
        perturbed,
        rules,
        clean_seeds=[11, 12, 13],
        perturbed_seeds=[11, 12, 13],
    )

    assert report.passed is True
    assert report.pair_count == 3
    assert report.seed_hash
    assert report.metrics["body_site_error"].relative_degradation == pytest.approx(0.02)
    assert report.metrics["coverage"].absolute_degradation == pytest.approx(0.01)

    bad = dict(perturbed)
    bad["early_termination"] = np.array([0.0, 0.0, 1.0])
    failed = compare_paired_metrics(clean, bad, rules)
    assert failed.passed is False
    assert failed.metrics["early_termination"].passed is False

    with pytest.raises(ValueError, match="same ordered seeds"):
        compare_paired_metrics(
            clean,
            perturbed,
            rules,
            clean_seeds=[11, 12, 13],
            perturbed_seeds=[13, 12, 11],
        )
