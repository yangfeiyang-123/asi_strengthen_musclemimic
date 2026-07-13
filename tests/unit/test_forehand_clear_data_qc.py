from __future__ import annotations

import numpy as np
import pytest

from musclemimic.badminton import data_qc


def _write_height_motion(source, cache, root_heights, *, right_hand_positions=None):
    root_heights = np.asarray(root_heights, dtype=np.float32)
    if right_hand_positions is None:
        right_hand_positions = np.zeros((root_heights.size, 3), np.float32)
    right_hand_positions = np.asarray(right_hand_positions, dtype=np.float32)
    np.savez(
        source,
        poses=np.zeros((60, 72), np.float32),
        mocap_framerate=60.0,
        mocap_frame_rate=60.0,
    )
    qpos = np.zeros((root_heights.size, 89), np.float32)
    qpos[:, 2] = root_heights
    joint_names = [
        "root",
        *[f"joint_{index}" for index in range(79)],
        "pro_sup_r",
        "deviation_r",
        "flexion_r",
    ]
    np.savez(
        cache,
        qpos=qpos,
        qvel=np.zeros((root_heights.size, 88), np.float32),
        site_xpos=right_hand_positions[:, None, :],
        site_xmat=np.broadcast_to(
            np.eye(3), (root_heights.size, 1, 3, 3)
        ).copy(),
        site_names=np.asarray(["right_hand_mimic"]),
        joint_names=np.asarray(joint_names),
        jnt_type=np.asarray([0, *([3] * 82)], dtype=np.int32),
        frequency=100.0,
    )


def test_qc_reports_missing_canonical_files_fail_closed(tmp_path):
    report = data_qc.inspect_canonical_dataset(tmp_path)
    assert report["passed"] is False
    assert len(report["hard_errors"]) == 27


def test_inspect_motion_distinguishes_contract_errors_from_spike_warnings(tmp_path):
    source = tmp_path / "source.npz"
    cache = tmp_path / "cache.npz"
    np.savez(source, poses=np.zeros((3, 72), np.float32), mocap_framerate=60.0, mocap_frame_rate=60.0)
    qpos = np.zeros((3, 89), np.float32)
    qpos[1, 0] = 0.05
    qpos[1, -1] = 0.50
    site = np.zeros((3, 1, 3), np.float32)
    site[1, 0, 0] = 0.25
    site_xmat = np.broadcast_to(np.eye(3), (3, 1, 3, 3)).copy()
    site_xmat[1, 0] = np.diag([1.0, -1.0, -1.0])
    joint_names = ["root", *[f"joint_{index}" for index in range(79)], "pro_sup_r", "deviation_r", "flexion_r"]
    np.savez(
        cache,
        qpos=qpos,
        qvel=np.zeros((3, 88), np.float32),
        site_xpos=site,
        site_xmat=site_xmat,
        site_names=np.asarray(["right_hand_mimic"]),
        joint_names=np.asarray(joint_names),
        jnt_type=np.asarray([0, *([3] * 82)], dtype=np.int32),
        frequency=100.0,
    )

    row, errors = data_qc._inspect_motion("m", source, cache)

    assert errors == []
    assert any("isolated root position jump" in warning for warning in row.warnings)
    assert any("right-hand speed spike" in warning for warning in row.warnings)
    assert any("right-wrist qpos step spike" in warning for warning in row.warnings)
    assert any("angular speed spike" in warning for warning in row.warnings)
    assert row.right_palm_proxy_site == "right_hand_mimic"
    assert row.max_right_palm_proxy_speed_m_s == row.max_right_hand_speed_m_s
    assert row.max_right_palm_proxy_angular_speed_rad_s == row.max_right_hand_angular_speed_rad_s
    assert row.max_racket_reference_angular_speed_rad_s == row.max_right_hand_angular_speed_rad_s
    assert row.cache_contains_racket_reference_site is False
    assert row.racket_reference_source == "derived_from_right_hand_rigid_attachment_at_runtime"


def test_inspect_motion_warns_on_material_source_cache_duration_misalignment(tmp_path):
    source = tmp_path / "source.npz"
    cache = tmp_path / "cache.npz"
    np.savez(
        source,
        poses=np.zeros((60, 72), np.float32),
        mocap_framerate=60.0,
        mocap_frame_rate=60.0,
    )
    joint_names = [
        "root",
        *[f"joint_{index}" for index in range(79)],
        "pro_sup_r",
        "deviation_r",
        "flexion_r",
    ]
    np.savez(
        cache,
        qpos=np.zeros((80, 89), np.float32),
        qvel=np.zeros((80, 88), np.float32),
        site_xpos=np.zeros((80, 1, 3), np.float32),
        site_xmat=np.broadcast_to(np.eye(3), (80, 1, 3, 3)),
        site_names=np.asarray(["right_hand_mimic"]),
        joint_names=np.asarray(joint_names),
        jnt_type=np.asarray([0, *([3] * 82)], dtype=np.int32),
        frequency=100.0,
    )

    row, errors = data_qc._inspect_motion("video1", source, cache)

    assert errors == []
    assert row.expected_cache_frames_from_source == 100
    assert row.cache_frame_alignment_delta == -20
    assert np.isclose(row.duration_alignment_error_s, 0.2)
    assert np.isclose(row.duration_alignment_error_fraction, 0.2)
    assert any("duration misalignment" in warning for warning in row.warnings)


def test_frequency_contract_keeps_source_ik_at_60_and_cache_control_at_100():
    assert data_qc.MAX_DURATION_ALIGNMENT_ERROR_S == 0.05
    # The report-level contract deliberately names both rates; target_fps in
    # Hydra remains the source/IK rate, not the extended control-cache rate.
    report = data_qc.inspect_canonical_dataset("does-not-exist")
    assert report["expected_source_fps"] == 60.0
    assert report["expected_cache_fps"] == 100.0
    assert "target_fps=60" in report["reference_contract"]["frequency_semantics"]


def test_joint_qpos_slices_handles_free_root_and_scalar_wrist_joints():
    names = ["root", "pro_sup_r", "deviation_r", "flexion_r"]
    slices = data_qc._joint_qpos_slices(
        names,
        np.asarray([0, 3, 3, 3], dtype=np.int32),
        10,
    )

    assert slices["root"] == (0, 7)
    assert slices["pro_sup_r"] == (7, 8)
    assert slices["flexion_r"] == (9, 10)


def test_isolated_jump_rejects_continuous_large_ramp_but_catches_one_frame_jump():
    ramp = np.zeros((6, 2), dtype=np.float64)
    ramp[:, 0] = 0.50
    jump = ramp.copy()
    jump[2, 1] = 0.90
    jump[3, 1] = -0.90

    assert data_qc._max_isolated_step(
        ramp,
        neighbor_change_threshold=0.25,
    ) == 0.0
    assert data_qc._max_isolated_step(
        jump,
        neighbor_change_threshold=0.25,
    ) == pytest.approx(0.90)


def test_isolated_jump_matches_video5_ramp_and_bad_5_bad_6_failure_shapes():
    # Measured raw_smooth_v1/video5 shoulder steps around its raw maximum:
    # entering change=0.2456, leaving change=0.0135.  It is a continuing ramp.
    video5 = np.asarray([[0.2137], [0.4593], [0.4458]], dtype=np.float64)
    # Measured pre-repair -5/-6 maxima change sharply on both sides.
    bad_5 = np.asarray([[0.4943], [0.9378], [0.2851]], dtype=np.float64)
    bad_6 = np.asarray([[0.5245], [1.2250], [0.4832]], dtype=np.float64)

    assert data_qc._max_isolated_step(
        video5, neighbor_change_threshold=0.25
    ) == 0.0
    assert data_qc._max_isolated_step(
        bad_5, neighbor_change_threshold=0.25
    ) == pytest.approx(0.9378)
    assert data_qc._max_isolated_step(
        bad_6, neighbor_change_threshold=0.25
    ) == pytest.approx(1.2250)


def test_root_jump_metric_allows_continuous_4_57m_s_but_catches_coordinate_reset():
    continuous = np.tile(np.asarray([[0.0457, 0.0, 0.0]]), (5, 1))
    reset = np.asarray(
        [
            [0.01, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [-0.08, 0.0, 0.0],
            [0.01, 0.0, 0.0],
        ]
    )

    assert data_qc._max_isolated_vector_step_speed(
        continuous,
        frequency=100.0,
        neighbor_velocity_change_threshold=2.0,
    ) == 0.0
    assert data_qc._max_isolated_vector_step_speed(
        reset,
        frequency=100.0,
        neighbor_velocity_change_threshold=2.0,
    ) == pytest.approx(10.0)
    assert data_qc.MAX_ABSOLUTE_ROOT_SPEED_M_S == 6.0


def test_persistent_low_root_warns_but_a_short_smooth_crouch_does_not(tmp_path):
    collapsed_source = tmp_path / "collapsed_source.npz"
    collapsed_cache = tmp_path / "collapsed_cache.npz"
    _write_height_motion(
        collapsed_source,
        collapsed_cache,
        np.full(100, 0.634, dtype=np.float32),
    )
    collapsed, errors = data_qc._inspect_motion(
        "6月2日-7", collapsed_source, collapsed_cache
    )

    assert errors == []
    assert collapsed.median_root_height_m == pytest.approx(0.634)
    assert collapsed.p10_root_height_m == pytest.approx(0.634)
    assert collapsed.min_root_height_m == pytest.approx(0.634)
    assert any(
        "persistent low-root/posture-collapse" in warning
        for warning in collapsed.warnings
    )

    crouch_source = tmp_path / "crouch_source.npz"
    crouch_cache = tmp_path / "crouch_cache.npz"
    short_crouch = np.concatenate(
        [
            np.linspace(0.95, 0.70, 10, endpoint=False),
            np.full(20, 0.70),
            np.linspace(0.70, 0.95, 10, endpoint=False),
            np.full(60, 0.95),
        ]
    )
    _write_height_motion(crouch_source, crouch_cache, short_crouch)
    crouch, errors = data_qc._inspect_motion(
        "normal_short_crouch", crouch_source, crouch_cache
    )

    assert errors == []
    assert crouch.min_root_height_m == pytest.approx(0.70)
    assert crouch.median_root_height_m == pytest.approx(0.95)
    assert not any(
        "persistent low-root/posture-collapse" in warning
        for warning in crouch.warnings
    )
    assert data_qc.MIN_MEDIAN_ROOT_HEIGHT_M == 0.85


def test_semantic_swing_gate_rejects_static_or_local_noise_but_accepts_full_swing(
    tmp_path,
):
    root_heights = np.full(100, 0.95, dtype=np.float32)

    static_source = tmp_path / "static_source.npz"
    static_cache = tmp_path / "static_cache.npz"
    _write_height_motion(static_source, static_cache, root_heights)
    static, errors = data_qc._inspect_motion(
        "static_forehand", static_source, static_cache
    )
    assert errors == []
    assert static.right_hand_path_length_m == pytest.approx(0.0)
    assert static.max_right_hand_displacement_m == pytest.approx(0.0)
    assert any(
        "insufficient forehand swing amplitude" in warning
        for warning in static.warnings
    )

    noise_source = tmp_path / "noise_source.npz"
    noise_cache = tmp_path / "noise_cache.npz"
    local_noise = np.zeros((100, 3), dtype=np.float32)
    local_noise[:, 0] = np.where(np.arange(100) % 2 == 0, -0.01, 0.01)
    _write_height_motion(
        noise_source,
        noise_cache,
        root_heights,
        right_hand_positions=local_noise,
    )
    noisy, errors = data_qc._inspect_motion("local_noise", noise_source, noise_cache)
    assert errors == []
    assert noisy.right_hand_path_length_m > data_qc.MIN_RIGHT_HAND_PATH_LENGTH_M
    assert noisy.max_right_hand_displacement_m < 0.03
    assert any(
        "insufficient forehand swing amplitude" in warning
        for warning in noisy.warnings
    )

    swing_source = tmp_path / "swing_source.npz"
    swing_cache = tmp_path / "swing_cache.npz"
    outward = np.linspace(0.0, 0.55, 50, endpoint=False)
    returning = np.linspace(0.55, 0.0, 50)
    full_swing = np.zeros((100, 3), dtype=np.float32)
    full_swing[:, 0] = np.concatenate((outward, returning))
    _write_height_motion(
        swing_source,
        swing_cache,
        root_heights,
        right_hand_positions=full_swing,
    )
    healthy, errors = data_qc._inspect_motion("full_swing", swing_source, swing_cache)
    assert errors == []
    assert healthy.right_hand_path_length_m >= 1.0
    assert healthy.max_right_hand_displacement_m >= 0.5
    assert not any(
        "insufficient forehand swing amplitude" in warning
        for warning in healthy.warnings
    )


def test_variant_paths_resolve_raw_smooth_namespace_and_reject_traversal(tmp_path):
    root, source, cache, suffix = data_qc._resolve_variant_paths(
        tmp_path,
        source_variant="raw_smooth_v1",
        cache_variant="raw_smooth_v1",
    )

    assert source == root / "temp" / "raw_smooth_v1"
    assert cache == root / "muscle_trajectory" / "raw_smooth_v1"
    assert suffix == ".npz"
    with pytest.raises(ValueError, match="safe namespace"):
        data_qc.inspect_canonical_dataset(
            tmp_path,
            source_variant="../raw",
            cache_variant="raw_smooth_v1",
        )


def test_qc_v3_reports_resolved_variants_and_clean_passed(tmp_path):
    report = data_qc.inspect_canonical_dataset(
        tmp_path,
        source_variant="raw_smooth_v1",
        cache_variant="raw_smooth_v1",
    )

    assert report["schema_version"] == "forehand_clear_data_qc_v3"
    assert report["source_variant"] == "raw_smooth_v1"
    assert report["cache_variant"] == "raw_smooth_v1"
    assert report["resolved_source_dir"].endswith("/temp/raw_smooth_v1")
    assert report["resolved_cache_dir"].endswith(
        "/muscle_trajectory/raw_smooth_v1"
    )
    assert report["passed"] is False
    assert report["clean_passed"] is False
