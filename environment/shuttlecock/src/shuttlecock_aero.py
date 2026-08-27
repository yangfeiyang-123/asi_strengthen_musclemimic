"""Custom aerodynamic model for a MuJoCo badminton shuttlecock.

Use this with assets/shuttlecock_mujoco.xml.

Recommended loop:
    data.qfrc_applied[:] = 0.0
    apply_shuttlecock_aero(model, data)
    mujoco.mj_step(model, data)

The model applies quadratic drag with a center of pressure behind the center of mass:
    F_D = -k |v_rel| v_rel,  k = m g / vt^2
This gives the desired terminal velocity vt. The pressure-center moment is folded into
a capped total torque and applied at the center of mass.

v2 extensions (all off at the legacy defaults, so archived rollouts are unchanged):
  - skirt cross-flow force ``F_N = -k * normal_force_gain * |v_rel| * v_perp``
    applied at the pressure center, where ``v_perp`` is the component of the
    relative wind perpendicular to the nose axis.  This adds the lateral force a
    yawed skirt really produces and stiffens the righting moment, instead of
    only inflating drag along ``-v`` as ``angle_drag_gain`` does;
  - anisotropic angular damping: the feather skirt damps tumbling far more than
    natural axial spin (``axial_spin_damping_nms_per_rad``, see the
    ``natural_spin_damping_nms_per_rad`` entry of params/shuttlecock_nominal.json);
  - an explicit ``wind_world_m_s`` override so domain randomization can perturb
    wind per episode without editing ``model.opt.wind``.

Use ``shuttlecock_aero_config_v2()`` for the tuned v2 preset and
``sample_randomized_aero_config()`` to draw a domain-randomized config from the
ranges in params/shuttlecock_nominal.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

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
    # v2 fields.  The defaults reproduce the legacy v1 forces exactly.
    normal_force_gain: float = 0.0
    axial_spin_damping_nms_per_rad: float | None = None
    wind_world_m_s: tuple[float, float, float] | None = None
    # Evaluate the aero force with the relative wind AT the pressure center
    # (v_com + omega x (cp - com)) instead of the COM velocity.  The rotating
    # skirt sweeping through air is what really damps the flip oscillation
    # ("fin damping"); steady nose-first descent has omega=0, so the calibrated
    # terminal velocity is unchanged.
    use_pressure_center_velocity: bool = False


@dataclass(frozen=True)
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
    force_torque_world_nm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    total_torque_world_nm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal_force_world_n: np.ndarray = field(default_factory=lambda: np.zeros(3))


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
    """Compute shuttlecock aerodynamic force, total torque, and diagnostics."""
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
    if cfg.use_pressure_center_velocity:
        v_rel = v_rel + np.cross(omega_world, cp_world - com_world)
    speed = float(np.linalg.norm(v_rel))
    if speed < 1e-8:
        force_world = np.zeros(3, dtype=float)
        damping_torque_world = np.zeros(3, dtype=float)
        force_torque_world = np.zeros(3, dtype=float)
        total_torque_world = np.zeros(3, dtype=float)
        diag = ShuttlecockAeroDiagnostics(
            speed_m_s=0.0,
            angle_of_attack_rad=0.0,
            drag_constant_kg_m=k,
            effective_drag_constant_kg_m=k,
            force_world_n=force_world.copy(),
            damping_torque_world_nm=damping_torque_world.copy(),
            center_of_pressure_world_m=cp_world.copy(),
            force_clipped=False,
            torque_clipped=False,
            force_torque_world_nm=force_torque_world.copy(),
            total_torque_world_nm=total_torque_world.copy(),
        )
        return force_world, total_torque_world, cp_world, diag

    v_hat = v_rel / speed
    cos_alpha = float(np.clip(np.dot(nose_axis_world, v_hat), -1.0, 1.0))
    angle_of_attack = float(np.arccos(cos_alpha))
    sin2_alpha = max(0.0, 1.0 - cos_alpha * cos_alpha)
    k_eff = k * (1.0 + cfg.angle_drag_gain * sin2_alpha)

    drag_force_world = -k_eff * speed * v_rel
    # Skirt cross-flow force: opposes the component of the relative wind that
    # is perpendicular to the nose axis.  Acting at the aft pressure center it
    # both drifts the shuttle laterally and rights the nose into the flow.
    v_perp = v_rel - float(np.dot(v_rel, nose_axis_world)) * nose_axis_world
    normal_force_world = -k * cfg.normal_force_gain * speed * v_perp
    force_world = drag_force_world + normal_force_world
    force_world, force_clipped = _clip_norm_with_flag(force_world, cfg.max_force_n)

    force_torque_world = np.cross(cp_world - com_world, force_world)
    if cfg.axial_spin_damping_nms_per_rad is None:
        damping_torque_world = -cfg.angular_damping_nms_per_rad * omega_world
    else:
        omega_axial = float(np.dot(omega_world, nose_axis_world)) * nose_axis_world
        omega_transverse = omega_world - omega_axial
        damping_torque_world = (
            -cfg.angular_damping_nms_per_rad * omega_transverse
            - cfg.axial_spin_damping_nms_per_rad * omega_axial
        )
    total_torque_world = force_torque_world + damping_torque_world
    total_torque_world, torque_clipped = _clip_norm_with_flag(
        total_torque_world,
        cfg.max_torque_nm,
    )

    diag = ShuttlecockAeroDiagnostics(
        speed_m_s=speed,
        angle_of_attack_rad=angle_of_attack,
        drag_constant_kg_m=k,
        effective_drag_constant_kg_m=k_eff,
        force_world_n=force_world.copy(),
        damping_torque_world_nm=damping_torque_world.copy(),
        center_of_pressure_world_m=cp_world.copy(),
        force_clipped=force_clipped,
        torque_clipped=torque_clipped,
        force_torque_world_nm=force_torque_world.copy(),
        total_torque_world_nm=total_torque_world.copy(),
        normal_force_world_n=normal_force_world.copy(),
    )
    return force_world, total_torque_world, cp_world, diag


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

    if cfg.wind_world_m_s is not None:
        wind = np.asarray(cfg.wind_world_m_s, dtype=float)
    elif cfg.use_model_wind:
        wind = np.array(model.opt.wind, dtype=float)
    else:
        wind = np.zeros(3)

    # Body local +Z in world coordinates.
    rot = np.array(data.xmat[body_id], dtype=float).reshape(3, 3)
    nose_axis_world = rot @ np.array([0.0, 0.0, 1.0])

    mass = float(model.body_mass[body_id])
    com_world = np.array(data.xipos[body_id], dtype=float)
    force_world, total_torque_world, _cp_world, diag = compute_shuttlecock_aero(
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
        total_torque_world,
        com_world,
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


NOMINAL_PARAMS_JSON_PATH = (
    Path(__file__).resolve().parents[1] / "params" / "shuttlecock_nominal.json"
)
SHUTTLE_DIAG_INERTIA_KG_M2 = (3.32e-6, 3.32e-6, 1.06e-6)


def restore_shuttle_inertia(
    model,
    body_name: str = "shuttle",
    inertia_diag_kg_m2: tuple[float, float, float] = SHUTTLE_DIAG_INERTIA_KG_M2,
) -> None:
    """Restore the physical shuttle inertia clamped away by ``boundinertia``.

    Scenes composed onto the MyoFullBody base spec inherit its
    ``<compiler boundinertia="0.0001">``, which silently raises the shuttle's
    diagonal inertia from ``(3.32e-6, 3.32e-6, 1.06e-6)`` to ``1e-4`` -- about
    30x -- making every rotational response (righting, cork angular impulse)
    30x too sluggish.  Call this once after loading such a composed model.

    Writing ``model.body_inertia`` alone is not enough on MuJoCo 3.4: the
    dynamics keep using precomputed inertia-derived constants until
    ``mj_setConst`` refreshes them, so it is called here with a throwaway
    ``MjData``.  Existing ``MjData`` instances remain valid afterwards.
    """
    _require_mujoco()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name!r}")
    model.body_inertia[body_id] = np.asarray(inertia_diag_kg_m2, dtype=float)
    mujoco.mj_setConst(model, mujoco.MjData(model))
V2_NORMAL_FORCE_GAIN = 1.5
V2_AXIAL_SPIN_DAMPING_NMS_PER_RAD = 5.0e-6


def shuttlecock_aero_config_v2(body_name: str = "shuttle", **overrides) -> ShuttlecockAeroConfig:
    """Return the tuned v2 aero preset (cross-flow force + anisotropic damping).

    The cross-flow gain 1.5 makes a fully sideways skirt roughly 2.7x as
    draggy as nose-first flight (1 + angle_drag_gain + normal_force_gain) and
    roughly triples the small-angle righting stiffness, which flips the shuttle
    nose-into-flow within a fraction of a second after a racket impact.  Axial
    spin decays with the far smaller natural spin damping from
    params/shuttlecock_nominal.json instead of the tumble damping.
    """
    cfg = ShuttlecockAeroConfig(
        body_name=body_name,
        normal_force_gain=V2_NORMAL_FORCE_GAIN,
        axial_spin_damping_nms_per_rad=V2_AXIAL_SPIN_DAMPING_NMS_PER_RAD,
        use_pressure_center_velocity=True,
    )
    return replace(cfg, **overrides) if overrides else cfg


def sample_randomized_aero_config(
    rng: np.random.Generator,
    *,
    base: ShuttlecockAeroConfig | None = None,
    params_json_path: str | Path | None = None,
    randomize_wind: bool = True,
) -> ShuttlecockAeroConfig:
    """Draw a domain-randomized aero config from params/shuttlecock_nominal.json.

    Samples terminal velocity, pressure-center offset, angle-of-attack drag
    gain, tumble damping, and (optionally) horizontal wind uniformly from the
    ``randomization`` ranges of the params file.  Mass and stringbed ranges in
    that block concern the MJCF model and the impact model and are not part of
    this config.  Fields without a published range (``normal_force_gain``,
    ``axial_spin_damping_nms_per_rad``) are inherited from ``base``.
    """
    path = Path(params_json_path) if params_json_path is not None else NOMINAL_PARAMS_JSON_PATH
    ranges = json.loads(path.read_text(encoding="utf-8"))["randomization"]

    def _uniform(name: str) -> float:
        low, high = (float(value) for value in ranges[name])
        if not (np.isfinite(low) and np.isfinite(high)) or high < low:
            raise ValueError(f"invalid randomization range for {name!r}: [{low}, {high}]")
        return float(rng.uniform(low, high))

    base_cfg = base if base is not None else shuttlecock_aero_config_v2()
    wind = base_cfg.wind_world_m_s
    if randomize_wind:
        wind = (_uniform("wind_m_s"), _uniform("wind_m_s"), 0.0)
    return replace(
        base_cfg,
        terminal_velocity_m_s=_uniform("terminal_velocity_m_s"),
        center_of_pressure_offset_m=_uniform("center_of_pressure_offset_m"),
        angle_drag_gain=_uniform("angle_drag_gain"),
        angular_damping_nms_per_rad=_uniform("angular_damping_nms_per_rad"),
        wind_world_m_s=wind,
    )
