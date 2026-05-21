import pytest

from musclemimic.utils.action_stage import MotionHints, classify_motion_stage


def test_stationary_visible_motion_is_base_candidate():
    metrics = {
        "reference_root_xy_total_displacement": 0.20,
        "reference_root_xy_peak_speed": 0.45,
        "reference_root_yaw_change": 0.10,
        "right_hand_world_path_length": 0.55,
    }

    result = classify_motion_stage(metrics, MotionHints(action_label="ForehandClear"))

    assert result.stage == "base"
    assert result.family == "general"
    assert "stationary_or_small_step" in result.reasons


def test_small_root_when_large_motion_expected_requires_repair():
    metrics = {
        "reference_root_xy_total_displacement": 0.12,
        "reference_root_xy_peak_speed": 0.40,
        "reference_root_yaw_change": 0.05,
        "right_hand_world_path_length": 0.60,
    }

    result = classify_motion_stage(
        metrics,
        MotionHints(action_label="ForehandNetLift", expected_large_motion=True),
    )

    assert result.stage == "repair"
    assert result.family == "net_frontcourt"
    assert "expected_large_motion_but_root_is_small" in result.reasons


def test_large_displacement_prefers_posttrain():
    metrics = {
        "reference_root_xy_total_displacement": 0.75,
        "reference_root_xy_peak_speed": 0.90,
        "reference_root_yaw_change": 0.20,
        "right_hand_world_path_length": 1.10,
    }

    result = classify_motion_stage(metrics, MotionHints(action_label="ForehandNetLift"))

    assert result.stage == "posttrain"
    assert result.family == "net_frontcourt"
    assert "large_root_displacement" in result.reasons


def test_high_speed_prefers_posttrain_even_with_medium_displacement():
    metrics = {
        "reference_root_xy_total_displacement": 0.42,
        "reference_root_xy_peak_speed": 1.35,
        "reference_root_yaw_change": 0.20,
        "right_hand_world_path_length": 1.00,
    }

    result = classify_motion_stage(metrics, MotionHints(action_label="Backhand"))

    assert result.stage == "posttrain"
    assert result.family == "rotation"
    assert "high_root_peak_speed" in result.reasons


def test_large_yaw_prefers_rotation_posttrain():
    metrics = {
        "reference_root_xy_total_displacement": 0.35,
        "reference_root_xy_peak_speed": 0.80,
        "reference_root_yaw_change": 1.10,
        "right_hand_world_path_length": 0.95,
    }

    result = classify_motion_stage(metrics, MotionHints(action_label="Smash"))

    assert result.stage == "posttrain"
    assert result.family == "rotation"
    assert "large_yaw_change" in result.reasons


def test_jump_or_lunge_hint_prefers_posttrain_family():
    metrics = {
        "reference_root_xy_total_displacement": 0.38,
        "reference_root_xy_peak_speed": 0.95,
        "reference_root_yaw_change": 0.25,
        "right_hand_world_path_length": 0.90,
    }

    result = classify_motion_stage(
        metrics,
        MotionHints(action_label="Smash", has_jump_or_lunge=True),
    )

    assert result.stage == "posttrain"
    assert result.family == "smash"
    assert "jump_or_lunge_hint" in result.reasons


def test_fine_hand_dominant_hint_excludes_motion():
    metrics = {
        "reference_root_xy_total_displacement": 0.08,
        "reference_root_xy_peak_speed": 0.25,
        "reference_root_yaw_change": 0.05,
        "right_hand_world_path_length": 0.25,
    }

    result = classify_motion_stage(
        metrics,
        MotionHints(action_label="NetTumble", fine_hand_dominant=True),
    )

    assert result.stage == "exclude"
    assert result.family == "fine_hand"
    assert "fine_hand_dominant" in result.reasons


def test_missing_required_metric_raises_clear_error():
    with pytest.raises(KeyError, match="reference_root_xy_total_displacement"):
        classify_motion_stage({}, MotionHints(action_label="ForehandClear"))
