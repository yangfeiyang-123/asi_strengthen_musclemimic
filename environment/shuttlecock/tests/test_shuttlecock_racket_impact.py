from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from src.shuttlecock_racket_impact import (
    ShuttlecockImpactConfig,
    compute_event_rebound,
    compute_event_rebound_velocity,
    set_freejoint_linear_velocity,
    should_apply_event_rebound,
)


def test_compute_event_rebound_velocity_reflects_closing_normal_component():
    cfg = ShuttlecockImpactConfig(event_restitution_normal=0.5, event_tangential_velocity_scale=0.8)

    result = compute_event_rebound_velocity(
        shuttle_velocity_world=np.array([0.0, 0.0, -10.0]),
        racket_surface_velocity_world=np.array([0.0, 0.0, 2.0]),
        normal_world=np.array([0.0, 0.0, 1.0]),
        cfg=cfg,
    )

    assert result == pytest.approx(np.array([0.0, 0.0, 8.0]))


def test_compute_event_rebound_velocity_preserves_scaled_tangential_component():
    cfg = ShuttlecockImpactConfig(event_restitution_normal=0.5, event_tangential_velocity_scale=0.8)

    result = compute_event_rebound_velocity(
        shuttle_velocity_world=np.array([3.0, 0.0, -10.0]),
        racket_surface_velocity_world=np.array([1.0, 0.0, 2.0]),
        normal_world=np.array([0.0, 0.0, 1.0]),
        cfg=cfg,
    )

    assert result == pytest.approx(np.array([2.6, 0.0, 8.0]))


def test_compute_event_rebound_velocity_returns_input_when_already_separating():
    cfg = ShuttlecockImpactConfig()
    velocity = np.array([0.0, 0.0, 4.0])

    result = compute_event_rebound_velocity(
        shuttle_velocity_world=velocity,
        racket_surface_velocity_world=np.zeros(3),
        normal_world=np.array([0.0, 0.0, 1.0]),
        cfg=cfg,
    )

    assert result == pytest.approx(velocity)


def test_compute_event_rebound_reports_clipping_diagnostics():
    cfg = ShuttlecockImpactConfig(max_rebound_speed_m_s=3.0)

    velocity, diag = compute_event_rebound(
        shuttle_velocity_world=np.array([0.0, 0.0, -10.0]),
        racket_surface_velocity_world=np.array([0.0, 0.0, 2.0]),
        normal_world=np.array([0.0, 0.0, 1.0]),
        cfg=cfg,
    )

    assert np.linalg.norm(velocity) == pytest.approx(3.0)
    assert diag.rebound_speed_m_s == pytest.approx(3.0)
    assert diag.rebound_clipped is True
    assert diag.relative_normal_velocity_m_s == pytest.approx(-12.0)


def test_impact_config_default_max_rebound_speed_matches_nominal_params():
    assert ShuttlecockImpactConfig().max_rebound_speed_m_s == pytest.approx(100.0)


def test_impact_config_default_tangential_velocity_scale_matches_nominal_params():
    assert ShuttlecockImpactConfig().event_tangential_velocity_scale == pytest.approx(0.85)


def test_should_apply_event_rebound_requires_active_fast_closing_contact():
    cfg = ShuttlecockImpactConfig(min_speed_for_event_m_s=5.0)

    assert should_apply_event_rebound({"active": True, "relative_normal_velocity": -5.1}, cfg) is True
    assert should_apply_event_rebound({"active": True, "relative_normal_velocity": -4.9}, cfg) is False
    assert should_apply_event_rebound({"active": False, "relative_normal_velocity": -10.0}, cfg) is False


@dataclass
class FakeModel:
    body_name_to_id: dict[str, int]
    body_jntadr: np.ndarray
    jnt_dofadr: np.ndarray
    jnt_type: np.ndarray


@dataclass
class FakeData:
    qvel: np.ndarray


def test_set_freejoint_linear_velocity_writes_first_three_freejoint_dofs():
    model = FakeModel(
        body_name_to_id={"shuttle": 0},
        body_jntadr=np.array([0]),
        jnt_dofadr=np.array([2]),
        jnt_type=np.array([0]),
    )
    data = FakeData(qvel=np.zeros(8))

    set_freejoint_linear_velocity(
        model,
        data,
        body_name="shuttle",
        velocity_world=np.array([1.0, 2.0, 3.0]),
        free_joint_type_value=0,
    )

    assert data.qvel.tolist() == [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
