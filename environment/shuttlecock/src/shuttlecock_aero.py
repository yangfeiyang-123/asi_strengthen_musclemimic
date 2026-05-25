"""Custom aerodynamic model for a MuJoCo badminton shuttlecock.

Use this with assets/shuttlecock_mujoco.xml.

Recommended loop:
    data.qfrc_applied[:] = 0.0
    apply_shuttlecock_aero(model, data)
    mujoco.mj_step(model, data)

The model applies quadratic drag at a center of pressure behind the center of mass:
    F_D = -k |v_rel| v_rel,  k = m g / vt^2
This gives the desired terminal velocity vt. Applying the force off-center generates the
head-first reorientation torque typical of a real shuttlecock.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    import mujoco
except Exception as exc:  # pragma: no cover
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None


@dataclass
class ShuttlecockAeroConfig:
    body_name: str = "shuttle"
    terminal_velocity_m_s: float = 6.86
    center_of_pressure_offset_m: float = 0.035
    angle_drag_gain: float = 0.20
    angular_damping_nms_per_rad: float = 2.0e-5
    max_force_n: float = 8.0
    max_torque_nm: float = 0.08
    use_model_wind: bool = True


@dataclass
class ShuttlecockAeroDiagnostics:
    speed_m_s: float
    angle_of_attack_rad: float
    drag_constant_kg_m: float
    effective_drag_constant_kg_m: float
    force_world_n: np.ndarray
    damping_torque_world_nm: np.ndarray
    center_of_pressure_world_m: np.ndarray
    force_clipped: bool
    torque_clipped: bool


def _require_mujoco() -> None:
    if mujoco is None:  # pragma: no cover
        raise RuntimeError(f"mujoco Python package is not available: {_MUJOCO_IMPORT_ERROR}")


def _clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > max_norm > 0:
        return vec * (max_norm / norm)
    return vec


def _clip_norm_with_flag(vec: np.ndarray, max_norm: float) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(vec))
    if norm > max_norm > 0:
        return vec * (max_norm / norm), True
    return vec, False


def compute_shuttlecock_aero(
    *,
    mass_kg: float,
    gravity: np.ndarray,
    wind: np.ndarray,
    v_world: np.ndarray,
    omega_world: np.ndarray,
    nose_axis_world: np.ndarray,
    com_world: np.ndarray,
    cfg: ShuttlecockAeroConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ShuttlecockAeroDiagnostics]:
    """Compute shuttlecock aerodynamic force, damping torque, and diagnostics."""
    gravity = np.asarray(gravity, dtype=float)
    wind = np.asarray(wind, dtype=float)
    v_world = np.asarray(v_world, dtype=float)
    omega_world = np.asarray(omega_world, dtype=float)
    nose_axis_world = np.asarray(nose_axis_world, dtype=float)
    com_world = np.asarray(com_world, dtype=float)

    nose_norm = float(np.linalg.norm(nose_axis_world))
    if nose_norm <= 1e-12:
        raise ValueError("nose_axis_world must be nonzero")
    nose_axis_world = nose_axis_world / nose_norm

    g = float(np.linalg.norm(gravity))
    if g <= 0:
        g = 9.81

    k = mass_kg * g / (cfg.terminal_velocity_m_s ** 2)
    cp_world = com_world - cfg.center_of_pressure_offset_m * nose_axis_world

    v_rel = v_world - wind
    speed = float(np.linalg.norm(v_rel))
    if speed < 1e-8:
        force_world = np.zeros(3, dtype=float)
        damping_torque_world = np.zeros(3, dtype=float)
        diag = ShuttlecockAeroDiagnostics(
            speed_m_s=speed,
            angle_of_attack_rad=0.0,
            drag_constant_kg_m=k,
            effective_drag_constant_kg_m=k,
            force_world_n=force_world,
            damping_torque_world_nm=damping_torque_world,
            center_of_pressure_world_m=cp_world,
            force_clipped=False,
            torque_clipped=False,
        )
        return force_world, damping_torque_world, cp_world, diag

    v_hat = v_rel / speed
    cos_alpha = float(np.clip(np.dot(nose_axis_world, v_hat), -1.0, 1.0))
    angle_of_attack = float(np.arccos(cos_alpha))
    sin2_alpha = max(0.0, 1.0 - cos_alpha * cos_alpha)
    k_eff = k * (1.0 + cfg.angle_drag_gain * sin2_alpha)

    force_world = -k_eff * speed * v_rel
    force_world, force_clipped = _clip_norm_with_flag(force_world, cfg.max_force_n)

    damping_torque_world = -cfg.angular_damping_nms_per_rad * omega_world
    damping_torque_world, torque_clipped = _clip_norm_with_flag(
        damping_torque_world,
        cfg.max_torque_nm,
    )

    diag = ShuttlecockAeroDiagnostics(
        speed_m_s=speed,
        angle_of_attack_rad=angle_of_attack,
        drag_constant_kg_m=k,
        effective_drag_constant_kg_m=k_eff,
        force_world_n=force_world,
        damping_torque_world_nm=damping_torque_world,
        center_of_pressure_world_m=cp_world,
        force_clipped=force_clipped,
        torque_clipped=torque_clipped,
    )
    return force_world, damping_torque_world, cp_world, diag


def apply_shuttlecock_aero(
    model,
    data,
    cfg: ShuttlecockAeroConfig | None = None,
) -> ShuttlecockAeroDiagnostics | None:
    """Apply shuttlecock aerodynamic force into data.qfrc_applied.

    Assumptions:
      - Body local +Z points toward cork/nose.
      - Body origin/inertial frame is at the center of mass.
      - Feathers are visual-only; only the cork geom collides.

    Notes:
      - mj_objectVelocity returns [angular_velocity, linear_velocity].
      - flg_local=0 returns world-oriented velocity components.
    """
    _require_mujoco()
    if cfg is None:
        cfg = ShuttlecockAeroConfig()

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {cfg.body_name!r}")

    vel6 = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel6, 0)
    omega_world = vel6[:3]
    v_world = vel6[3:]

    wind = np.array(model.opt.wind, dtype=float) if cfg.use_model_wind else np.zeros(3)

    # Body local +Z in world coordinates.
    rot = np.array(data.xmat[body_id], dtype=float).reshape(3, 3)
    nose_axis_world = rot @ np.array([0.0, 0.0, 1.0])

    mass = float(model.body_mass[body_id])
    com_world = np.array(data.xipos[body_id], dtype=float)
    force_world, damping_torque_world, cp_world, diag = compute_shuttlecock_aero(
        mass_kg=mass,
        gravity=np.array(model.opt.gravity, dtype=float),
        wind=wind,
        v_world=v_world,
        omega_world=omega_world,
        nose_axis_world=nose_axis_world,
        com_world=com_world,
        cfg=cfg,
    )

    mujoco.mj_applyFT(
        model,
        data,
        force_world,
        damping_torque_world,
        cp_world,
        body_id,
        data.qfrc_applied,
    )
    return diag


def expected_drag_constant(mass_kg: float = 0.00519, g_m_s2: float = 9.81, vt_m_s: float = 6.86) -> float:
    """Return k in F=-k|v|v for the selected terminal velocity."""
    return mass_kg * g_m_s2 / (vt_m_s ** 2)


def equivalent_cd(k: float, rho: float = 1.225, tip_diameter_m: float = 0.065) -> float:
    """Convert k to a nominal Cd using tip-circle area A=pi*(D/2)^2."""
    area = np.pi * (tip_diameter_m / 2.0) ** 2
    return 2.0 * k / (rho * area)
