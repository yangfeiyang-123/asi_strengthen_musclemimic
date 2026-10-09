"""Tests for the v2 shuttlecock aero extensions.

Covers: legacy defaults staying bit-exact, the skirt cross-flow force, the
anisotropic angular damping split, pressure-center-velocity ("fin") damping,
the wind override, domain randomization, and two integration-level checks
(sideways righting, high-clear trajectory plausibility).
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from environment.shuttlecock.src.shuttlecock_aero import (
    NOMINAL_PARAMS_JSON_PATH,
    ShuttlecockAeroConfig,
    apply_shuttlecock_aero,
    compute_shuttlecock_aero,
    sample_randomized_aero_config,
    shuttlecock_aero_config_v2,
)

SHUTTLE_XML = Path(__file__).resolve().parents[1] / "assets" / "shuttlecock_mujoco.xml"


def _aero(cfg: ShuttlecockAeroConfig, *, v, omega, nose, wind=None):
    return compute_shuttlecock_aero(
        mass_kg=0.00519,
        gravity=np.array([0.0, 0.0, -9.81]),
        wind=np.zeros(3) if wind is None else np.asarray(wind, dtype=float),
        v_world=np.asarray(v, dtype=float),
        omega_world=np.asarray(omega, dtype=float),
        nose_axis_world=np.asarray(nose, dtype=float),
        com_world=np.zeros(3),
        cfg=cfg,
    )


def test_legacy_defaults_are_unchanged_by_v2_fields():
    cfg = ShuttlecockAeroConfig()
    assert cfg.normal_force_gain == 0.0
    assert cfg.axial_spin_damping_nms_per_rad is None
    assert cfg.wind_world_m_s is None
    assert cfg.use_pressure_center_velocity is False

    v = np.array([9.0, -2.0, 4.0])
    omega = np.array([3.0, 7.0, -1.0])
    nose = np.array([0.3, 0.1, 0.95])
    nose /= np.linalg.norm(nose)
    force, torque, cp, diag = _aero(cfg, v=v, omega=omega, nose=nose)

    speed = float(np.linalg.norm(v))
    k = 0.00519 * 9.81 / cfg.terminal_velocity_m_s**2
    cos_alpha = float(np.dot(nose, v / speed))
    k_eff = k * (1.0 + cfg.angle_drag_gain * (1.0 - cos_alpha**2))
    expected_force = -k_eff * speed * v
    expected_torque = np.cross(cp, expected_force) - cfg.angular_damping_nms_per_rad * omega
    np.testing.assert_allclose(force, expected_force, rtol=0, atol=1e-15)
    np.testing.assert_allclose(torque, expected_torque, rtol=0, atol=1e-15)
    np.testing.assert_allclose(diag.normal_force_world_n, np.zeros(3), atol=0)


def test_cross_flow_force_adds_lateral_component_and_righting_torque():
    v = np.array([10.0, 0.0, 0.0])
    nose = np.array([0.0, 0.0, 1.0])  # fully sideways: alpha = 90 deg
    base = ShuttlecockAeroConfig(use_model_wind=False)
    v2 = shuttlecock_aero_config_v2(use_pressure_center_velocity=False)

    force_base, torque_base, _, _ = _aero(base, v=v, omega=np.zeros(3), nose=nose)
    force_v2, torque_v2, _, diag_v2 = _aero(v2, v=v, omega=np.zeros(3), nose=nose)

    # The cross-flow term opposes v_perp (= v here), so total drag grows.
    assert float(np.linalg.norm(force_v2)) > float(np.linalg.norm(force_base)) * 1.5
    assert float(np.dot(diag_v2.normal_force_world_n, v)) < 0.0
    # Torque about +y rotates the +z nose toward the +x flow: righting.
    assert float(torque_v2[1]) > float(torque_base[1]) > 0.0


def test_anisotropic_damping_splits_axial_and_transverse():
    v2 = shuttlecock_aero_config_v2(use_pressure_center_velocity=False)
    nose = np.array([0.0, 0.0, 1.0])
    v = 5.0 * nose  # aligned flight so cp torque vanishes

    axial = _aero(v2, v=v, omega=10.0 * nose, nose=nose)[3]
    np.testing.assert_allclose(
        axial.damping_torque_world_nm,
        -v2.axial_spin_damping_nms_per_rad * 10.0 * nose,
        atol=1e-12,
    )
    transverse_omega = np.array([10.0, 0.0, 0.0])
    transverse = _aero(v2, v=v, omega=transverse_omega, nose=nose)[3]
    np.testing.assert_allclose(
        transverse.damping_torque_world_nm,
        -v2.angular_damping_nms_per_rad * transverse_omega,
        atol=1e-12,
    )
    assert v2.axial_spin_damping_nms_per_rad < v2.angular_damping_nms_per_rad


def test_pressure_center_velocity_damps_tumbling():
    """With the fin-damping term, a tumbling aligned shuttle feels an opposing torque."""
    nose = np.array([0.0, 0.0, 1.0])
    v = 12.0 * nose
    omega = np.array([40.0, 0.0, 0.0])  # transverse tumble
    off = shuttlecock_aero_config_v2(
        use_pressure_center_velocity=False, angular_damping_nms_per_rad=0.0
    )
    on = shuttlecock_aero_config_v2(
        use_pressure_center_velocity=True, angular_damping_nms_per_rad=0.0
    )
    torque_off = _aero(off, v=v, omega=omega, nose=nose)[1]
    torque_on = _aero(on, v=v, omega=omega, nose=nose)[1]
    assert float(np.dot(torque_on, omega)) < -1.0e-5
    assert float(np.dot(torque_on, omega)) < float(np.dot(torque_off, omega))


def test_wind_override_takes_precedence_over_model_wind():
    model = mujoco.MjModel.from_xml_path(str(SHUTTLE_XML))
    data = mujoco.MjData(model)
    data.qvel[0:3] = [10.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    override = ShuttlecockAeroConfig(wind_world_m_s=(3.0, 0.0, 0.0))
    diag = apply_shuttlecock_aero(model, data, override)
    assert diag.speed_m_s == pytest.approx(7.0)

    legacy = ShuttlecockAeroConfig()
    diag_legacy = apply_shuttlecock_aero(model, data, legacy)
    assert diag_legacy.speed_m_s == pytest.approx(10.0)  # model wind is zero


def test_randomized_config_samples_within_published_ranges():
    ranges = json.loads(NOMINAL_PARAMS_JSON_PATH.read_text(encoding="utf-8"))["randomization"]
    rng = np.random.default_rng(7)
    for _ in range(20):
        cfg = sample_randomized_aero_config(rng)
        assert ranges["terminal_velocity_m_s"][0] <= cfg.terminal_velocity_m_s <= ranges["terminal_velocity_m_s"][1]
        assert (
            ranges["center_of_pressure_offset_m"][0]
            <= cfg.center_of_pressure_offset_m
            <= ranges["center_of_pressure_offset_m"][1]
        )
        assert ranges["angle_drag_gain"][0] <= cfg.angle_drag_gain <= ranges["angle_drag_gain"][1]
        assert (
            ranges["angular_damping_nms_per_rad"][0]
            <= cfg.angular_damping_nms_per_rad
            <= ranges["angular_damping_nms_per_rad"][1]
        )
        assert cfg.wind_world_m_s is not None
        assert all(
            ranges["wind_m_s"][0] <= w <= ranges["wind_m_s"][1] for w in cfg.wind_world_m_s[:2]
        )
        assert cfg.wind_world_m_s[2] == 0.0
        # v2 structure fields are inherited from the default v2 base
        assert cfg.normal_force_gain == shuttlecock_aero_config_v2().normal_force_gain
        assert cfg.use_pressure_center_velocity is True

    a = sample_randomized_aero_config(np.random.default_rng(3))
    b = sample_randomized_aero_config(np.random.default_rng(3))
    assert a == b


def _fly(cfg: ShuttlecockAeroConfig, *, pos, vel, quat, max_steps=8000):
    model = mujoco.MjModel.from_xml_path(str(SHUTTLE_XML))
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    data = mujoco.MjData(model)
    data.qpos[0:3] = pos
    data.qpos[3:7] = quat
    data.qvel[0:3] = vel
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shuttle")
    rows = []
    for step in range(max_steps):
        data.qfrc_applied[:] = 0.0
        apply_shuttlecock_aero(model, data, cfg)
        mujoco.mj_step(model, data)
        rot = np.array(data.xmat[body], dtype=float).reshape(3, 3)
        rows.append(
            (
                float(data.time),
                *np.array(data.qpos[0:3], dtype=float),
                *np.array(data.qvel[0:3], dtype=float),
                *rot[:, 2],
            )
        )
        if float(data.qpos[2]) <= 0.05 and step > 100:
            break
    return np.asarray(rows), float(model.opt.timestep)


def test_v2_rights_a_sideways_shuttle_much_faster_than_legacy():
    quat = np.array([1.0, 0.0, 0.0, 0.0])  # nose +z, velocity +x: alpha = 90 deg
    pos = [0.0, 0.0, 3.0]
    vel = [15.0, 0.0, 0.0]

    def max_alpha_deg(rows, dt, t0, t1):
        i0, i1 = int(t0 / dt), int(t1 / dt)
        window = rows[i0:i1]
        v = window[:, 4:7]
        nose = window[:, 7:10]
        v_hat = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        cos = np.clip(np.sum(nose * v_hat, axis=1), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)).max())

    rows_v2, dt = _fly(shuttlecock_aero_config_v2(), pos=pos, vel=vel, quat=quat, max_steps=1200)
    rows_v1, _ = _fly(ShuttlecockAeroConfig(), pos=pos, vel=vel, quat=quat, max_steps=1200)
    # Measured: v2 envelope < 6 deg in [0.3, 0.4] s; legacy still oscillates > 40 deg.
    assert max_alpha_deg(rows_v2, dt, 0.3, 0.4) < 15.0
    assert max_alpha_deg(rows_v1, dt, 0.3, 0.4) > 25.0


def test_v2_high_clear_trajectory_is_plausible():
    """Launch a full clear from the backcourt; check range, apex, and asymmetry."""
    speed, elevation = 28.0, np.deg2rad(42.0)
    vel = np.array([speed * np.cos(elevation), 0.0, speed * np.sin(elevation)])
    nose = vel / np.linalg.norm(vel)
    # nose aligned with launch velocity
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, nose)
    axis /= np.linalg.norm(axis)
    angle = float(np.arccos(np.clip(np.dot(z_axis, nose), -1.0, 1.0)))
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, axis, angle)

    rows, _dt = _fly(shuttlecock_aero_config_v2(), pos=[-4.6, 0.0, 2.0], vel=vel, quat=quat)

    landing = rows[-1]
    apex_index = int(np.argmax(rows[:, 3]))
    horizontal_range = float(landing[1] - rows[0, 1])
    apex_height = float(rows[apex_index, 3])
    landing_speed = float(np.linalg.norm(landing[4:7]))
    ascent_horizontal = float(rows[apex_index, 1] - rows[0, 1])
    descent_horizontal = float(landing[1] - rows[apex_index, 1])

    assert 8.0 <= horizontal_range <= 11.0
    assert 4.5 <= apex_height <= 7.5
    assert 5.5 <= landing_speed <= 7.5  # lands near terminal velocity
    # Shuttle trajectories are strongly asymmetric: steep drag-braked descent.
    assert ascent_horizontal > 1.4 * descent_horizontal
