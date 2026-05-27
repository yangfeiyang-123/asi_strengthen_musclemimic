from __future__ import annotations

import numpy as np
import pytest

from environment.overall_environment.src.impact_target import (
    BodyScale,
    ImpactTargetConfig,
    extract_impact_target_from_sites,
    regularize_impact_target,
)


def test_extract_impact_target_prefers_peak_virtual_racket_speed():
    right_hand_pos = np.array(
        [
            [0.20, 0.20, 1.40],
            [0.25, 0.25, 1.55],
            [0.35, 0.32, 1.75],
            [0.48, 0.36, 1.88],
            [0.55, 0.34, 1.82],
        ],
        dtype=float,
    )
    root_pos = np.zeros_like(right_hand_pos)
    forward_axis = np.tile(np.array([1.0, 0.0, 0.0]), (right_hand_pos.shape[0], 1))
    right_axis = np.tile(np.array([0.0, 1.0, 0.0]), (right_hand_pos.shape[0], 1))

    target = extract_impact_target_from_sites(
        right_hand_pos=right_hand_pos,
        root_pos=root_pos,
        forward_axis=forward_axis,
        right_axis=right_axis,
        dt=0.01,
        racket_length_m=0.67,
    )

    assert target.impact_frame == 3
    assert 0.0 < target.impact_phase < 1.0
    assert target.position_root_local[0] > 0.9
    assert target.position_root_local[1] > 0.30
    assert target.position_root_local[2] == pytest.approx(1.88)
    np.testing.assert_allclose(np.linalg.norm(target.racket_head_velocity_dir), 1.0)


def test_regularize_impact_target_clamps_to_forehand_comfort_zone():
    raw = np.array([0.05, -0.20, 2.80], dtype=float)
    scale = BodyScale(
        shoulder_height_m=1.42,
        arm_reach_up_m=0.72,
        racket_effective_length_m=0.52,
    )
    cfg = ImpactTargetConfig(
        min_forward_offset_m=0.28,
        max_forward_offset_m=0.85,
        min_racket_side_offset_m=0.18,
        max_racket_side_offset_m=0.65,
        reach_alpha=0.78,
        racket_beta=0.82,
        min_height_margin_m=-0.08,
        max_height_margin_m=0.06,
    )

    result = regularize_impact_target(raw, scale, cfg)

    assert result[0] == pytest.approx(0.28)
    assert result[1] == pytest.approx(0.18)
    expected_height = 1.42 + 0.72 * 0.78 + 0.52 * 0.82 + 0.06
    assert result[2] == pytest.approx(expected_height)
