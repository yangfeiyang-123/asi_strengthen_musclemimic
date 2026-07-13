from __future__ import annotations

import numpy as np
import pytest

from environment.shuttlecock.src.shuttlecock_aero import (
    ShuttlecockAeroConfig,
    compute_shuttlecock_aero,
    equivalent_cd,
    expected_drag_constant,
)


def test_expected_drag_constant_and_equivalent_cd_match_nominal_design():
    k = expected_drag_constant()

    assert k == pytest.approx(0.0010819, rel=5e-4)
    assert equivalent_cd(k) == pytest.approx(0.532, rel=5e-3)


def test_drag_force_opposes_relative_velocity_and_reports_diagnostics():
    cfg = ShuttlecockAeroConfig()
    force, torque, cp, diag = compute_shuttlecock_aero(
        mass_kg=0.00519,
        gravity=np.array([0.0, 0.0, -9.81]),
        wind=np.zeros(3),
        v_world=np.array([10.0, 0.0, 0.0]),
        omega_world=np.zeros(3),
        nose_axis_world=np.array([1.0, 0.0, 0.0]),
        com_world=np.array([0.0, 0.0, 1.0]),
        cfg=cfg,
    )

    assert np.dot(force, np.array([10.0, 0.0, 0.0])) < 0.0
    assert diag.speed_m_s == pytest.approx(10.0)
    assert diag.angle_of_attack_rad == pytest.approx(0.0)
    assert diag.force_clipped is False
    assert diag.torque_clipped is False
    assert cp == pytest.approx(np.array([-0.035, 0.0, 1.0]))
    assert torque == pytest.approx(np.zeros(3))


def test_sideways_flight_increases_effective_drag_constant():
    cfg = ShuttlecockAeroConfig(angle_drag_gain=0.5)
    aligned = compute_shuttlecock_aero(
        mass_kg=0.00519,
        gravity=np.array([0.0, 0.0, -9.81]),
        wind=np.zeros(3),
        v_world=np.array([12.0, 0.0, 0.0]),
        omega_world=np.zeros(3),
        nose_axis_world=np.array([1.0, 0.0, 0.0]),
        com_world=np.zeros(3),
        cfg=cfg,
    )[3]
    sideways = compute_shuttlecock_aero(
        mass_kg=0.00519,
        gravity=np.array([0.0, 0.0, -9.81]),
        wind=np.zeros(3),
        v_world=np.array([12.0, 0.0, 0.0]),
        omega_world=np.zeros(3),
        nose_axis_world=np.array([0.0, 0.0, 1.0]),
        com_world=np.zeros(3),
        cfg=cfg,
    )[3]

    assert sideways.effective_drag_constant_kg_m > aligned.effective_drag_constant_kg_m
    assert sideways.angle_of_attack_rad == pytest.approx(np.pi / 2.0)


def test_force_and_torque_clipping_are_reported():
    cfg = ShuttlecockAeroConfig(max_force_n=0.01, max_torque_nm=0.001)
    force, torque, _cp, diag = compute_shuttlecock_aero(
        mass_kg=0.00519,
        gravity=np.array([0.0, 0.0, -9.81]),
        wind=np.zeros(3),
        v_world=np.array([100.0, 0.0, 0.0]),
        omega_world=np.array([100.0, 0.0, 0.0]),
        nose_axis_world=np.array([1.0, 0.0, 0.0]),
        com_world=np.zeros(3),
        cfg=cfg,
    )

    assert np.linalg.norm(force) == pytest.approx(0.01)
    assert np.linalg.norm(torque) == pytest.approx(0.001)
    assert diag.force_clipped is True
    assert diag.torque_clipped is True


def test_total_aero_torque_from_pressure_center_is_clipped():
    cfg = ShuttlecockAeroConfig(
        max_force_n=8.0,
        max_torque_nm=0.08,
        angular_damping_nms_per_rad=0.0,
    )
    force, torque, cp, diag = compute_shuttlecock_aero(
        mass_kg=0.00519,
        gravity=np.array([0.0, 0.0, -9.81]),
        wind=np.zeros(3),
        v_world=np.array([100.0, 0.0, 0.0]),
        omega_world=np.zeros(3),
        nose_axis_world=np.array([0.0, 0.0, 1.0]),
        com_world=np.zeros(3),
        cfg=cfg,
    )

    raw_cp_torque = np.cross(cp - np.zeros(3), force)
    assert np.linalg.norm(raw_cp_torque) > cfg.max_torque_nm
    assert np.linalg.norm(torque) == pytest.approx(cfg.max_torque_nm)
    assert diag.torque_clipped is True


def test_near_zero_speed_returns_zero_force_and_zero_speed_diagnostics():
    force, torque, _cp, diag = compute_shuttlecock_aero(
        mass_kg=0.00519,
        gravity=np.array([0.0, 0.0, -9.81]),
        wind=np.zeros(3),
        v_world=np.array([1e-9, 0.0, 0.0]),
        omega_world=np.array([1.0, 0.0, 0.0]),
        nose_axis_world=np.array([1.0, 0.0, 0.0]),
        com_world=np.zeros(3),
        cfg=ShuttlecockAeroConfig(),
    )

    assert force == pytest.approx(np.zeros(3))
    assert torque == pytest.approx(np.zeros(3))
    assert diag.speed_m_s == 0.0
    assert diag.force_world_n == pytest.approx(np.zeros(3))
    assert diag.damping_torque_world_nm == pytest.approx(np.zeros(3))


def test_diagnostics_arrays_do_not_alias_returned_arrays():
    force, torque, cp, diag = compute_shuttlecock_aero(
        mass_kg=0.00519,
        gravity=np.array([0.0, 0.0, -9.81]),
        wind=np.zeros(3),
        v_world=np.array([10.0, 0.0, 0.0]),
        omega_world=np.array([1.0, 0.0, 0.0]),
        nose_axis_world=np.array([1.0, 0.0, 0.0]),
        com_world=np.zeros(3),
        cfg=ShuttlecockAeroConfig(),
    )

    force[:] = 123.0
    torque[:] = 456.0
    cp[:] = 789.0

    assert not np.all(diag.force_world_n == 123.0)
    assert not np.all(diag.damping_torque_world_nm == 456.0)
    assert not np.all(diag.center_of_pressure_world_m == 789.0)
