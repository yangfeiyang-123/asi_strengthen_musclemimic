"""Offline shuttle feed-trajectory generator for the incoming-hit task.

The legacy ``rejection_v2`` path retains its pure-NumPy, instant-righting
model byte-for-byte.  ``calibrated_rigid_body_v3`` instead integrates the same
free rigid body, orientation, inertia and aerodynamic torque used online by
MuJoCo.  That distinction is deliberate: a high clear can accumulate a large
position error when an instant-righting trajectory is used as the label for a
rotating online shuttle.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from environment.shuttlecock.src.shuttlecock_aero import (
    ShuttlecockAeroConfig,
    compute_shuttlecock_aero,
)

SHUTTLE_MASS_KG = 0.00519
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)
GROUND_REST_HEIGHT_M = 0.035

FEED_BANK_MANIFEST_SCHEMA = "incoming_shuttle_feed_bank_manifest_v1"
FEED_BANK_GENERATOR = {
    "name": "environment.overall_environment.src.shuttle_feeder.build_feed_bank",
    "version": "2",
}
CALIBRATED_FEED_BANK_GENERATOR = {
    "name": "environment.overall_environment.src.shuttle_feeder.build_feed_bank",
    "version": "3.3-online-euler-cork-schema",
}


class FeedBankValidationError(ValueError):
    """Raised when a persisted feed bank does not match its exact provenance."""


@dataclass(frozen=True)
class HitWindow:
    """Axis-aligned box in front of and above the player where a hit is possible."""

    x_range: tuple[float, float] = (-3.05, -2.35)
    y_range: tuple[float, float] = (-0.55, 0.55)
    z_range: tuple[float, float] = (1.85, 2.25)

    def __post_init__(self) -> None:
        for name in ("x_range", "y_range", "z_range"):
            _validate_finite_range(name, getattr(self, name))

    def contains(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        inside = (
            (points[:, 0] >= self.x_range[0])
            & (points[:, 0] <= self.x_range[1])
            & (points[:, 1] >= self.y_range[0])
            & (points[:, 1] <= self.y_range[1])
            & (points[:, 2] >= self.z_range[0])
            & (points[:, 2] <= self.z_range[1])
        )
        return inside

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [
                0.5 * (self.x_range[0] + self.x_range[1]),
                0.5 * (self.y_range[0] + self.y_range[1]),
                0.5 * (self.z_range[0] + self.z_range[1]),
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class FeedConfig:
    """Physical and semantic constraints for a reachable overhead feed.

    The intercept constraints are part of the generated-data ABI.  They keep
    low emergency pickups, still-rising shuttles and feeds that begin the base
    swing before reset out of Stage 3 rather than asking PPO to learn around
    bad task data.
    """

    launch_x_range: tuple[float, float] = (3.0, 5.8)
    launch_y_range: tuple[float, float] = (-0.8, 0.8)
    launch_z_range: tuple[float, float] = (1.7, 2.2)
    speed_range: tuple[float, float] = (13.0, 18.0)
    elevation_deg_range: tuple[float, float] = (24.0, 40.0)
    azimuth_jitter_deg: float = 6.0
    integration_dt: float = 0.002
    max_flight_time: float = 3.0
    max_attempts: int = 600
    net_x: float = 0.0
    net_clearance_height: float = 1.75
    intercept_time_range_s: tuple[float, float] = (1.05, 1.45)
    intercept_vertical_velocity_range_m_s: tuple[float, float] = (-5.5, -1.0)
    apex_height_range_m: tuple[float, float] = (2.4, 4.2)
    target_height_tolerance_m: float = 0.08
    # Opt-in Stage-3 contact-acquisition generator.  These fields are omitted
    # from rejection_v2 manifests so existing v2 artifacts keep their exact
    # producer contract and are never overwritten merely by upgrading code.
    sampling_mode: str = "rejection_v2"
    calibrated_launch_pos: tuple[float, float, float] = (3.0, 0.0, 2.0)
    calibrated_launch_vel: tuple[float, float, float] = (
        -9.448524702570104,
        -0.20954003865622678,
        17.76046320587704,
    )
    calibrated_intercept_time_s: float = 1.88
    calibrated_intercept_fraction_jitter: float = 0.42
    calibrated_reference_seed: int = 17
    calibrated_warmup_count: int = 16
    calibrated_position_jitter_m: tuple[float, float, float] = (0.15, 0.14, 0.16)
    calibrated_velocity_jitter_m_s: tuple[float, float, float] = (0.55, 0.35, 1.10)

    def __post_init__(self) -> None:
        for name in (
            "launch_x_range",
            "launch_y_range",
            "launch_z_range",
            "speed_range",
            "elevation_deg_range",
            "intercept_time_range_s",
            "intercept_vertical_velocity_range_m_s",
            "apex_height_range_m",
        ):
            _validate_finite_range(name, getattr(self, name))
        if self.speed_range[0] <= 0.0:
            raise ValueError("speed_range must be strictly positive")
        if not -89.0 < self.elevation_deg_range[0] < self.elevation_deg_range[1] < 89.0:
            raise ValueError("elevation_deg_range must lie strictly inside (-89, 89)")
        if self.intercept_time_range_s[0] < 0.0:
            raise ValueError("intercept_time_range_s must be non-negative")
        if self.intercept_vertical_velocity_range_m_s[1] >= 0.0:
            raise ValueError("intercept_vertical_velocity_range_m_s must be descending")
        for name in ("integration_dt", "max_flight_time", "target_height_tolerance_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.max_attempts, bool) or int(self.max_attempts) != self.max_attempts:
            raise ValueError("max_attempts must be a positive integer")
        if int(self.max_attempts) <= 0:
            raise ValueError("max_attempts must be a positive integer")
        for name in ("azimuth_jitter_deg", "net_x", "net_clearance_height"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.azimuth_jitter_deg < 0.0:
            raise ValueError("azimuth_jitter_deg must be non-negative")
        if self.net_clearance_height < 0.0:
            raise ValueError("net_clearance_height must be non-negative")
        if self.sampling_mode not in {"rejection_v2", "calibrated_rigid_body_v3"}:
            raise ValueError(
                "sampling_mode must be rejection_v2 or calibrated_rigid_body_v3"
            )
        for name in (
            "calibrated_launch_pos",
            "calibrated_launch_vel",
            "calibrated_position_jitter_m",
            "calibrated_velocity_jitter_m_s",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain three finite numbers")
        if np.any(np.asarray(self.calibrated_position_jitter_m, dtype=float) < 0.0):
            raise ValueError("calibrated_position_jitter_m must be non-negative")
        if np.any(np.asarray(self.calibrated_velocity_jitter_m_s, dtype=float) < 0.0):
            raise ValueError("calibrated_velocity_jitter_m_s must be non-negative")
        if not math.isfinite(float(self.calibrated_intercept_time_s)) or float(
            self.calibrated_intercept_time_s
        ) <= 0.0:
            raise ValueError("calibrated_intercept_time_s must be finite and positive")
        if not math.isfinite(float(self.calibrated_intercept_fraction_jitter)) or not (
            0.0 <= float(self.calibrated_intercept_fraction_jitter) <= 0.5
        ):
            raise ValueError(
                "calibrated_intercept_fraction_jitter must be finite and in [0, 0.5]"
            )
        if self.sampling_mode == "calibrated_rigid_body_v3":
            if (
                float(self.calibrated_intercept_time_s)
                < self.intercept_time_range_s[0]
                or float(self.calibrated_intercept_time_s)
                > self.intercept_time_range_s[1]
            ):
                raise ValueError(
                    "calibrated_intercept_time_s must remain inside "
                    "intercept_time_range_s"
                )
        for name in ("calibrated_reference_seed", "calibrated_warmup_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class FeedSample:
    launch_pos: np.ndarray
    launch_vel: np.ndarray
    trajectory: np.ndarray  # (T, 7): [t, x, y, z, vx, vy, vz]
    intercept_index: int
    intercept_point: np.ndarray
    intercept_velocity: np.ndarray
    intercept_time_s: float
    # Optional physical point used for contact semantics.  Legacy v2 banks
    # omit this field and continue to use the body-COM trajectory byte-for-byte.
    # Rigid-body v3 banks store the cork-site trajectory here because racket
    # contact occurs at the cork, not at the shuttle body's free-joint origin.
    contact_trajectory: np.ndarray | None = None


def _validate_finite_range(name: str, value: tuple[float, float]) -> None:
    try:
        low, high = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly two finite numbers") from exc
    if not math.isfinite(low) or not math.isfinite(high) or not low < high:
        raise ValueError(f"{name} must be a strictly increasing finite range")


def integrate_shuttle_flight(
    pos0: np.ndarray,
    vel0: np.ndarray,
    *,
    dt: float = 0.002,
    t_max: float = 3.0,
    aero_cfg: ShuttlecockAeroConfig | None = None,
    mass_kg: float = SHUTTLE_MASS_KG,
    gravity: np.ndarray = GRAVITY,
    ground_height: float = GROUND_REST_HEIGHT_M,
) -> np.ndarray:
    """Semi-implicit Euler integration of the drag-affected flight.

    Returns an (T, 7) array of [t, x, y, z, vx, vy, vz], stopping at t_max or
    when the shuttle reaches ground rest height.
    """
    if aero_cfg is None:
        aero_cfg = ShuttlecockAeroConfig()
    pos = np.asarray(pos0, dtype=float).copy()
    vel = np.asarray(vel0, dtype=float).copy()
    gravity = np.asarray(gravity, dtype=float)
    rows = [np.concatenate([[0.0], pos, vel])]
    steps = int(np.ceil(t_max / dt))
    for step in range(1, steps + 1):
        speed = float(np.linalg.norm(vel))
        if speed > 1e-8:
            nose_axis = vel / speed  # righted shuttle: nose tracks the flow
        else:
            nose_axis = np.array([0.0, 0.0, 1.0])
        force, _torque, _cp, _diag = compute_shuttlecock_aero(
            mass_kg=mass_kg,
            gravity=gravity,
            wind=np.zeros(3),
            v_world=vel,
            omega_world=np.zeros(3),
            nose_axis_world=nose_axis,
            com_world=pos,
            cfg=aero_cfg,
        )
        vel = vel + dt * (force / mass_kg + gravity)
        pos = pos + dt * vel
        rows.append(np.concatenate([[step * dt], pos, vel]))
        if pos[2] <= ground_height:
            break
    return np.asarray(rows, dtype=float)


def launch_quat_from_velocity(vel: np.ndarray) -> np.ndarray:
    """Return a wxyz quaternion aligning the shuttle nose (body +Z) with the flow.

    A flying shuttle points its cork/nose along the velocity direction, so body
    +Z must map to v_hat. This keeps the offline zero-angle-of-attack assumption
    consistent with the online aero model at launch.
    """
    vel = np.asarray(vel, dtype=float)
    speed = float(np.linalg.norm(vel))
    if speed <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    target = vel / speed
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z_axis, target), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if dot < -1.0 + 1e-12:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    axis = np.cross(z_axis, target)
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * np.arccos(dot)
    return np.concatenate([[np.cos(half)], np.sin(half) * axis]).astype(float)


def net_crossing_height(trajectory: np.ndarray, cfg: FeedConfig) -> float | None:
    """Return the interpolated first right-to-left net-crossing height."""

    x = trajectory[:, 1]
    z = trajectory[:, 3]
    crossing = np.nonzero((x[:-1] > cfg.net_x) & (x[1:] <= cfg.net_x))[0]
    if crossing.size == 0:
        return None
    i = int(crossing[0])
    frac = (x[i] - cfg.net_x) / max(x[i] - x[i + 1], 1e-12)
    return float(z[i] + frac * (z[i + 1] - z[i]))


def _crosses_net_cleanly(trajectory: np.ndarray, cfg: FeedConfig) -> bool:
    height = net_crossing_height(trajectory, cfg)
    return height is not None and height > cfg.net_clearance_height


def sample_feed(
    rng: np.random.Generator,
    cfg: FeedConfig | None = None,
    window: HitWindow | None = None,
    aero_cfg: ShuttlecockAeroConfig | None = None,
) -> FeedSample:
    """Rejection-sample one physically safe, descending overhead feed.

    A target point is sampled before each launch.  Among valid trajectory rows
    we select the point closest to the target *height* (with a small lateral
    tie-break), rather than the old first row entering the window.  The old
    rule systematically selected a boundary and produced a harmful low tail.
    """
    if cfg is None:
        cfg = FeedConfig()
    if window is None:
        window = HitWindow()
    if aero_cfg is None:
        aero_cfg = ShuttlecockAeroConfig()

    rejected = {
        "net": 0,
        "apex": 0,
        "window_time_velocity": 0,
        "target_height": 0,
    }
    for _ in range(cfg.max_attempts):
        target_point = np.array(
            [
                rng.uniform(*window.x_range),
                rng.uniform(*window.y_range),
                rng.uniform(*window.z_range),
            ],
            dtype=float,
        )
        launch_pos = np.array(
            [
                rng.uniform(*cfg.launch_x_range),
                rng.uniform(*cfg.launch_y_range),
                rng.uniform(*cfg.launch_z_range),
            ],
            dtype=float,
        )
        speed = rng.uniform(*cfg.speed_range)
        elevation = np.deg2rad(rng.uniform(*cfg.elevation_deg_range))
        to_target = target_point - launch_pos
        base_azimuth = float(np.arctan2(to_target[1], to_target[0]))
        azimuth = base_azimuth + np.deg2rad(rng.uniform(-cfg.azimuth_jitter_deg, cfg.azimuth_jitter_deg))
        launch_vel = speed * np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ],
            dtype=float,
        )
        trajectory = integrate_shuttle_flight(
            launch_pos,
            launch_vel,
            dt=cfg.integration_dt,
            t_max=cfg.max_flight_time,
            aero_cfg=aero_cfg,
        )
        if not _crosses_net_cleanly(trajectory, cfg):
            rejected["net"] += 1
            continue
        apex_height = float(np.max(trajectory[:, 3]))
        if not cfg.apex_height_range_m[0] <= apex_height <= cfg.apex_height_range_m[1]:
            rejected["apex"] += 1
            continue
        inside = window.contains(trajectory[:, 1:4])
        approaching = trajectory[:, 4] < 0.0
        in_time = (
            (trajectory[:, 0] >= cfg.intercept_time_range_s[0])
            & (trajectory[:, 0] <= cfg.intercept_time_range_s[1])
        )
        descending = (
            (trajectory[:, 6] >= cfg.intercept_vertical_velocity_range_m_s[0])
            & (trajectory[:, 6] <= cfg.intercept_vertical_velocity_range_m_s[1])
        )
        candidates = np.nonzero(inside & approaching & in_time & descending)[0]
        if candidates.size == 0:
            rejected["window_time_velocity"] += 1
            continue
        height_error = np.abs(trajectory[candidates, 3] - target_point[2])
        lateral_error = np.abs(trajectory[candidates, 2] - target_point[1])
        score = height_error / (window.z_range[1] - window.z_range[0])
        score += 0.25 * lateral_error / (window.y_range[1] - window.y_range[0])
        selected = int(np.argmin(score))
        if height_error[selected] > cfg.target_height_tolerance_m:
            rejected["target_height"] += 1
            continue
        intercept_index = int(candidates[selected])
        return FeedSample(
            launch_pos=launch_pos,
            launch_vel=launch_vel,
            trajectory=trajectory,
            intercept_index=intercept_index,
            intercept_point=trajectory[intercept_index, 1:4].copy(),
            intercept_velocity=trajectory[intercept_index, 4:7].copy(),
            intercept_time_s=float(trajectory[intercept_index, 0]),
        )
    raise RuntimeError(
        "failed to sample a quality-controlled overhead feed after "
        f"{cfg.max_attempts} attempts; rejection_counts={rejected}"
    )


class _RigidBodyFlightIntegrator:
    """Small exact online-dynamics replica used only while generating v3 data."""

    def __init__(self, *, dt: float, t_max: float) -> None:
        import mujoco

        from environment.shuttlecock.src.shuttlecock_aero import (
            ShuttlecockAeroConfig,
        )

        asset = (
            Path(__file__).resolve().parents[2]
            / "shuttlecock"
            / "assets"
            / "shuttlecock_mujoco.xml"
        )
        if not asset.is_file():
            raise FileNotFoundError(f"rigid-body shuttle asset is missing: {asset}")
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(asset))
        self.model.opt.timestep = float(dt)
        # The production full-body scene leaves MuJoCo's integrator at Euler,
        # while the standalone shuttle demo requests implicitfast.  At a
        # 20 m/s launch this difference moves the shuttle by >10 cm at impact.
        # Generated feeds must reproduce the online environment exactly.
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        self.data = mujoco.MjData(self.model)
        joint = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "shuttle_free",
        )
        if joint < 0:
            raise ValueError("rigid-body shuttle asset has no shuttle_free joint")
        self.qadr = int(self.model.jnt_qposadr[joint])
        self.dadr = int(self.model.jnt_dofadr[joint])
        self.body_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "shuttle")
        )
        self.cork_site_id = int(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "cork_contact_site"
            )
        )
        if self.body_id < 0 or self.cork_site_id < 0:
            raise ValueError("rigid-body shuttle asset is missing body/cork site")
        self.aero_cfg = ShuttlecockAeroConfig(body_name="shuttle")
        self.dt = float(dt)
        self.t_max = float(t_max)

    def _cork_velocity(self) -> np.ndarray:
        velocity = np.zeros(6, dtype=float)
        self._mujoco.mj_objectVelocity(
            self.model,
            self.data,
            self._mujoco.mjtObj.mjOBJ_BODY,
            self.body_id,
            velocity,
            0,
        )
        omega, origin_velocity = velocity[:3], velocity[3:]
        cork = np.asarray(self.data.site_xpos[self.cork_site_id], dtype=float)
        origin = np.asarray(self.data.xpos[self.body_id], dtype=float)
        return origin_velocity + np.cross(omega, cork - origin)

    def integrate(
        self, launch_pos: np.ndarray, launch_vel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        from environment.shuttlecock.src.shuttlecock_aero import (
            apply_shuttlecock_aero,
        )

        mujoco = self._mujoco
        position = np.asarray(launch_pos, dtype=float)
        velocity = np.asarray(launch_vel, dtype=float)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr : self.qadr + 3] = position
        self.data.qpos[self.qadr + 3 : self.qadr + 7] = launch_quat_from_velocity(
            velocity
        )
        self.data.qvel[self.dadr : self.dadr + 3] = velocity
        self.data.qvel[self.dadr + 3 : self.dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        rows = [np.concatenate([[0.0], position, velocity])]
        cork_rows = [
            np.concatenate(
                [
                    [0.0],
                    np.asarray(
                        self.data.site_xpos[self.cork_site_id], dtype=float
                    ),
                    self._cork_velocity(),
                ]
            )
        ]
        for step in range(1, int(math.ceil(self.t_max / self.dt)) + 1):
            self.data.qfrc_applied[:] = 0.0
            apply_shuttlecock_aero(self.model, self.data, self.aero_cfg)
            mujoco.mj_step(self.model, self.data)
            position = np.asarray(
                self.data.qpos[self.qadr : self.qadr + 3], dtype=float
            ).copy()
            velocity = np.asarray(
                self.data.qvel[self.dadr : self.dadr + 3], dtype=float
            ).copy()
            rows.append(np.concatenate([[step * self.dt], position, velocity]))
            cork_rows.append(
                np.concatenate(
                    [
                        [step * self.dt],
                        np.asarray(
                            self.data.site_xpos[self.cork_site_id], dtype=float
                        ),
                        self._cork_velocity(),
                    ]
                )
            )
            if position[2] <= GROUND_REST_HEIGHT_M:
                break
        return np.asarray(rows, dtype=float), np.asarray(cork_rows, dtype=float)


def _valid_intercept_candidates(
    trajectory: np.ndarray,
    cfg: FeedConfig,
    window: HitWindow,
) -> np.ndarray:
    inside = window.contains(trajectory[:, 1:4])
    return np.nonzero(
        inside
        & (trajectory[:, 4] < 0.0)
        & (trajectory[:, 0] >= cfg.intercept_time_range_s[0])
        & (trajectory[:, 0] <= cfg.intercept_time_range_s[1])
        & (trajectory[:, 6] >= cfg.intercept_vertical_velocity_range_m_s[0])
        & (trajectory[:, 6] <= cfg.intercept_vertical_velocity_range_m_s[1])
    )[0]


def _calibrated_sample(
    *,
    rng: np.random.Generator,
    cfg: FeedConfig,
    window: HitWindow,
    integrator: _RigidBodyFlightIntegrator,
    jitter_scale: float,
    target_intercept_fraction: float | None,
) -> FeedSample:
    center_pos = np.asarray(cfg.calibrated_launch_pos, dtype=float)
    center_vel = np.asarray(cfg.calibrated_launch_vel, dtype=float)
    position_jitter = np.asarray(cfg.calibrated_position_jitter_m, dtype=float)
    velocity_jitter = np.asarray(cfg.calibrated_velocity_jitter_m_s, dtype=float)
    rejected = {
        "launch_contract": 0,
        "net_or_apex": 0,
        "intercept": 0,
    }
    for _ in range(cfg.max_attempts):
        if jitter_scale == 0.0:
            launch_pos = center_pos.copy()
            launch_vel = center_vel.copy()
        else:
            launch_pos = center_pos + rng.uniform(-1.0, 1.0, 3) * position_jitter * jitter_scale
            launch_vel = center_vel + rng.uniform(-1.0, 1.0, 3) * velocity_jitter * jitter_scale
        launch_speed = float(np.linalg.norm(launch_vel))
        launch_elevation = float(
            np.rad2deg(
                np.arctan2(launch_vel[2], np.linalg.norm(launch_vel[:2]))
            )
        )
        if not (
            cfg.launch_x_range[0] <= launch_pos[0] <= cfg.launch_x_range[1]
            and cfg.launch_y_range[0] <= launch_pos[1] <= cfg.launch_y_range[1]
            and cfg.launch_z_range[0] <= launch_pos[2] <= cfg.launch_z_range[1]
            and cfg.speed_range[0] <= launch_speed <= cfg.speed_range[1]
            and cfg.elevation_deg_range[0]
            <= launch_elevation
            <= cfg.elevation_deg_range[1]
        ):
            rejected["launch_contract"] += 1
            continue
        trajectory, cork_trajectory = integrator.integrate(launch_pos, launch_vel)
        net_height = net_crossing_height(trajectory, cfg)
        apex = float(np.max(trajectory[:, 3]))
        if (
            net_height is None
            or net_height <= cfg.net_clearance_height
            or not cfg.apex_height_range_m[0]
            <= apex
            <= cfg.apex_height_range_m[1]
        ):
            rejected["net_or_apex"] += 1
            continue
        candidates = _valid_intercept_candidates(cork_trajectory, cfg, window)
        if candidates.size == 0:
            rejected["intercept"] += 1
            continue
        boundary_band = max(0.01, 0.05 * (window.z_range[1] - window.z_range[0]))
        interior = candidates[
            (cork_trajectory[candidates, 3] > window.z_range[0] + boundary_band)
            & (cork_trajectory[candidates, 3] < window.z_range[1] - boundary_band)
        ]
        if interior.size == 0:
            rejected["intercept"] += 1
            continue
        candidates = interior
        point_scale = np.maximum(
            np.array(
                [
                    window.x_range[1] - window.x_range[0],
                    window.y_range[1] - window.y_range[0],
                    window.z_range[1] - window.z_range[0],
                ],
                dtype=float,
            ),
            1.0e-6,
        )
        point_error = np.linalg.norm(
            (cork_trajectory[candidates, 1:4] - window.center) / point_scale,
            axis=1,
        )
        time_span = max(
            cfg.intercept_time_range_s[1] - cfg.intercept_time_range_s[0],
            1.0e-6,
        )
        if target_intercept_fraction is None:
            target_time = float(cfg.calibrated_intercept_time_s)
        else:
            candidate_times = cork_trajectory[candidates, 0]
            target_time = float(
                candidate_times[0]
                + float(target_intercept_fraction)
                * (candidate_times[-1] - candidate_times[0])
            )
        time_error = np.abs(cork_trajectory[candidates, 0] - target_time) / time_span
        # The frozen racket follows a time-parameterized sweep through this
        # window.  Time therefore defines reachability; the small spatial term
        # only breaks ties and must not collapse every sample back to the box
        # centre (the old false-diversity failure mode).
        intercept_index = int(
            candidates[int(np.argmin(time_error + 0.02 * point_error))]
        )
        return FeedSample(
            launch_pos=launch_pos,
            launch_vel=launch_vel,
            trajectory=trajectory,
            intercept_index=intercept_index,
            intercept_point=cork_trajectory[intercept_index, 1:4].copy(),
            intercept_velocity=cork_trajectory[intercept_index, 4:7].copy(),
            intercept_time_s=float(cork_trajectory[intercept_index, 0]),
            contact_trajectory=cork_trajectory,
        )
    raise RuntimeError(
        "failed to sample a calibrated rigid-body feed after "
        f"{cfg.max_attempts} attempts; rejection_counts={rejected}"
    )


def _build_calibrated_feed_bank(
    n: int,
    seed: int,
    cfg: FeedConfig,
    window: HitWindow,
) -> list[FeedSample]:
    if int(n) <= 0:
        raise ValueError("feed bank size must be positive")
    rng = np.random.default_rng(seed)
    integrator = _RigidBodyFlightIntegrator(
        dt=cfg.integration_dt,
        t_max=cfg.max_flight_time,
    )
    bank: list[FeedSample] = []
    reference_bank = int(seed) == int(cfg.calibrated_reference_seed)
    for index in range(int(n)):
        exact_reference = reference_bank and index == 0
        if exact_reference:
            scale = 0.0
        elif not reference_bank:
            # Curriculum ordering belongs only to the canonical training bank.
            # Held-out seeds must cover the full perturbation distribution;
            # otherwise evaluation silently measures only the easy curriculum.
            scale = 1.0
        elif index < int(cfg.calibrated_warmup_count):
            scale = 0.15
        elif index < max(2 * int(cfg.calibrated_warmup_count), 32):
            scale = 0.45
        else:
            scale = 1.0
        target_intercept_fraction = None
        if scale > 0.0:
            target_intercept_fraction = float(
                0.5
                + rng.uniform(-1.0, 1.0)
                * cfg.calibrated_intercept_fraction_jitter
                * scale
            )
        bank.append(
            _calibrated_sample(
                rng=rng,
                cfg=cfg,
                window=window,
                integrator=integrator,
                jitter_scale=scale,
                target_intercept_fraction=target_intercept_fraction,
            )
        )
    return bank


def build_feed_bank(
    n: int,
    seed: int,
    cfg: FeedConfig | None = None,
    window: HitWindow | None = None,
    aero_cfg: ShuttlecockAeroConfig | None = None,
) -> list[FeedSample]:
    cfg = FeedConfig() if cfg is None else cfg
    window = HitWindow() if window is None else window
    if cfg.sampling_mode == "calibrated_rigid_body_v3":
        return _build_calibrated_feed_bank(int(n), int(seed), cfg, window)
    rng = np.random.default_rng(seed)
    return [sample_feed(rng, cfg, window, aero_cfg) for _ in range(int(n))]


def feed_bank_quality_report(
    bank: list[FeedSample],
    cfg: FeedConfig | None = None,
    window: HitWindow | None = None,
) -> dict[str, Any]:
    """Compute fail-closed semantic QC for a persisted or generated bank."""

    if not bank:
        raise ValueError("feed-bank quality report requires at least one sample")
    cfg = FeedConfig() if cfg is None else cfg
    window = HitWindow() if window is None else window
    for sample in bank:
        _validate_feed_sample(sample)

    launch_positions = np.stack(
        [np.asarray(sample.launch_pos, dtype=float) for sample in bank]
    )
    launch_velocities = np.stack(
        [np.asarray(sample.launch_vel, dtype=float) for sample in bank]
    )
    launch_speeds = np.linalg.norm(launch_velocities, axis=1)
    launch_elevations_deg = np.rad2deg(
        np.arctan2(
            launch_velocities[:, 2],
            np.linalg.norm(launch_velocities[:, :2], axis=1),
        )
    )
    points = np.stack([np.asarray(sample.intercept_point, dtype=float) for sample in bank])
    velocities = np.stack(
        [np.asarray(sample.intercept_velocity, dtype=float) for sample in bank]
    )
    times = np.asarray([sample.intercept_time_s for sample in bank], dtype=float)
    apex_heights = np.asarray(
        [float(np.max(sample.trajectory[:, 3])) for sample in bank], dtype=float
    )
    net_heights = np.asarray(
        [
            float("nan")
            if (height := net_crossing_height(sample.trajectory, cfg)) is None
            else height
            for sample in bank
        ],
        dtype=float,
    )
    heights = points[:, 2]
    height_quantiles = np.quantile(heights, [0.0, 0.05, 0.5, 0.95, 1.0])
    window_height_span = window.z_range[1] - window.z_range[0]
    height_coverage_fraction = float(
        (height_quantiles[3] - height_quantiles[1]) / window_height_span
    )
    boundary_band = max(0.01, 0.05 * window_height_span)
    boundary_fraction = float(
        np.mean(
            (heights <= window.z_range[0] + boundary_band)
            | (heights >= window.z_range[1] - boundary_band)
        )
    )
    distribution_applicable = len(bank) >= 32
    launch_position_valid = np.ones(len(bank), dtype=bool)
    for axis, bounds in enumerate(
        (cfg.launch_x_range, cfg.launch_y_range, cfg.launch_z_range)
    ):
        launch_position_valid &= (
            (launch_positions[:, axis] >= bounds[0])
            & (launch_positions[:, axis] <= bounds[1])
        )

    gates = {
        "all_launch_positions_valid": bool(launch_position_valid.all()),
        "all_launch_speeds_valid": bool(
            (
                (launch_speeds >= cfg.speed_range[0])
                & (launch_speeds <= cfg.speed_range[1])
            ).all()
        ),
        "all_launch_elevations_valid": bool(
            (
                (launch_elevations_deg >= cfg.elevation_deg_range[0])
                & (launch_elevations_deg <= cfg.elevation_deg_range[1])
            ).all()
        ),
        "all_in_window": bool(window.contains(points).all()),
        "all_approaching_player": bool((velocities[:, 0] < 0.0).all()),
        "all_intercept_times_valid": bool(
            (
                (times >= cfg.intercept_time_range_s[0])
                & (times <= cfg.intercept_time_range_s[1])
            ).all()
        ),
        "all_intercepts_descending": bool(
            (
                (velocities[:, 2] >= cfg.intercept_vertical_velocity_range_m_s[0])
                & (velocities[:, 2] <= cfg.intercept_vertical_velocity_range_m_s[1])
            ).all()
        ),
        "all_apex_heights_valid": bool(
            (
                (apex_heights >= cfg.apex_height_range_m[0])
                & (apex_heights <= cfg.apex_height_range_m[1])
            ).all()
        ),
        "all_net_crossings_clear": bool(
            np.isfinite(net_heights).all()
            and (net_heights > cfg.net_clearance_height).all()
        ),
        "intercept_height_coverage": bool(
            not distribution_applicable or height_coverage_fraction >= 0.65
        ),
        "no_intercept_height_boundary_pileup": bool(
            not distribution_applicable or boundary_fraction <= 0.25
        ),
    }
    return {
        "schema_version": "incoming_shuttle_feed_quality_v2",
        "sample_count": len(bank),
        "distribution_gates_applicable": distribution_applicable,
        "gates": gates,
        "passed": bool(all(gates.values())),
        "launch_position_quantiles_m": _vector_quantiles(launch_positions),
        "launch_speed_quantiles_m_s": _scalar_quantiles(launch_speeds),
        "launch_elevation_quantiles_deg": _scalar_quantiles(
            launch_elevations_deg
        ),
        "intercept_point_quantiles": _vector_quantiles(points),
        "intercept_time_quantiles_s": _scalar_quantiles(times),
        "intercept_speed_quantiles_m_s": _scalar_quantiles(
            np.linalg.norm(velocities, axis=1)
        ),
        "intercept_vertical_velocity_quantiles_m_s": _scalar_quantiles(
            velocities[:, 2]
        ),
        "apex_height_quantiles_m": _scalar_quantiles(apex_heights),
        "net_crossing_height_quantiles_m": _scalar_quantiles(net_heights),
        "intercept_height_coverage_fraction_p05_p95": height_coverage_fraction,
        "intercept_height_boundary_fraction": boundary_fraction,
        "intercept_height_boundary_band_m": boundary_band,
    }


def render_feed_bank_qc(
    path: str | Path,
    bank: list[FeedSample],
    cfg: FeedConfig | None = None,
    window: HitWindow | None = None,
    *,
    title: str = "Overhead shuttle feed QC",
    max_trajectories: int = 24,
) -> Path:
    """Render trajectory geometry and intercept distributions to one PNG."""

    if not bank:
        raise ValueError("cannot render an empty feed bank")
    cfg = FeedConfig() if cfg is None else cfg
    window = HitWindow() if window is None else window
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = min(max(1, int(max_trajectories)), len(bank))
    ordered = np.argsort([sample.intercept_point[2] for sample in bank])
    selected = ordered[np.linspace(0, len(ordered) - 1, count).round().astype(int)]
    points = np.stack([sample.intercept_point for sample in bank])
    times = np.asarray([sample.intercept_time_s for sample in bank])

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    side, top, height_hist, time_hist = axes.ravel()
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, count))
    for color, index in zip(colors, selected):
        sample = bank[int(index)]
        side.plot(sample.trajectory[:, 1], sample.trajectory[:, 3], color=color, alpha=0.55)
        top.plot(sample.trajectory[:, 1], sample.trajectory[:, 2], color=color, alpha=0.55)
    side.scatter(points[:, 0], points[:, 2], s=12, color="tab:red", alpha=0.75, label="intercept")
    side.axvline(cfg.net_x, color="black", linewidth=1.0)
    side.plot(
        [cfg.net_x, cfg.net_x],
        [0.0, cfg.net_clearance_height],
        color="black",
        linewidth=4.0,
        label="minimum net clearance",
    )
    side.add_patch(
        Rectangle(
            (window.x_range[0], window.z_range[0]),
            window.x_range[1] - window.x_range[0],
            window.z_range[1] - window.z_range[0],
            fill=False,
            edgecolor="tab:red",
            linewidth=2.0,
        )
    )
    side.set(xlabel="court x (m)", ylabel="height z (m)", title="Side view")
    side.legend(loc="best")
    side.grid(alpha=0.2)

    top.scatter(points[:, 0], points[:, 1], s=12, color="tab:red", alpha=0.75)
    top.add_patch(
        Rectangle(
            (window.x_range[0], window.y_range[0]),
            window.x_range[1] - window.x_range[0],
            window.y_range[1] - window.y_range[0],
            fill=False,
            edgecolor="tab:red",
            linewidth=2.0,
        )
    )
    top.axvline(cfg.net_x, color="black", linewidth=1.5)
    top.set(xlabel="court x (m)", ylabel="court y (m)", title="Top view")
    top.grid(alpha=0.2)

    height_hist.hist(points[:, 2], bins=min(20, max(6, len(bank) // 8)), color="tab:blue")
    height_hist.axvspan(*window.z_range, color="tab:green", alpha=0.12)
    height_hist.set(xlabel="intercept height (m)", ylabel="count", title="Intercept height")
    height_hist.grid(alpha=0.2)

    time_hist.hist(times, bins=min(20, max(6, len(bank) // 8)), color="tab:orange")
    time_hist.axvspan(*cfg.intercept_time_range_s, color="tab:green", alpha=0.12)
    time_hist.set(xlabel="intercept time (s)", ylabel="count", title="Reaction time")
    time_hist.grid(alpha=0.2)
    figure.suptitle(title)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def _scalar_quantiles(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(np.asarray(values, dtype=float), [0.0, 0.05, 0.5, 0.95, 1.0])
    return {
        name: float(value)
        for name, value in zip(("min", "p05", "median", "p95", "max"), quantiles)
    }


def _vector_quantiles(values: np.ndarray) -> dict[str, list[float]]:
    quantiles = np.quantile(
        np.asarray(values, dtype=float), [0.0, 0.05, 0.5, 0.95, 1.0], axis=0
    )
    return {
        name: [float(item) for item in row]
        for name, row in zip(("min", "p05", "median", "p95", "max"), quantiles)
    }


def feed_bank_manifest_path(path: str | Path) -> Path:
    """Return the sidecar path without changing the bank's portable filename."""
    value = Path(path)
    return value.with_suffix(value.suffix + ".manifest.json")


def feed_bank_contract(
    *,
    seed: int,
    sample_count: int,
    cfg: FeedConfig | None = None,
    window: HitWindow | None = None,
    aero_cfg: ShuttlecockAeroConfig | None = None,
) -> dict[str, Any]:
    """Describe every input that deterministically defines a generated bank."""
    if int(sample_count) <= 0:
        raise ValueError("feed bank sample_count must be positive")
    resolved_cfg = FeedConfig() if cfg is None else cfg
    calibrated = resolved_cfg.sampling_mode == "calibrated_rigid_body_v3"
    config_payload = asdict(resolved_cfg)
    if not calibrated:
        # Preserve the exact v2 producer ABI.  New opt-in calibration fields
        # must not invalidate or trigger replacement of existing v2 banks.
        config_payload.pop("sampling_mode", None)
        for key in tuple(config_payload):
            if key.startswith("calibrated_"):
                config_payload.pop(key)
    return {
        "schema_version": FEED_BANK_MANIFEST_SCHEMA,
        "generator": dict(
            CALIBRATED_FEED_BANK_GENERATOR if calibrated else FEED_BANK_GENERATOR
        ),
        "seed": int(seed),
        "sample_count": int(sample_count),
        "feed_config": _json_value(config_payload),
        "hit_window": _json_value(HitWindow() if window is None else window),
        "aero_config": _json_value(
            ShuttlecockAeroConfig() if aero_cfg is None else aero_cfg
        ),
    }


def feed_sample_fingerprint(sample: FeedSample) -> str:
    """Hash the exact semantic sample content independent of NPZ compression."""
    _validate_feed_sample(sample)
    digest = hashlib.sha256()
    for label, value in (
        ("launch_pos", sample.launch_pos),
        ("launch_vel", sample.launch_vel),
        ("trajectory", sample.trajectory),
    ):
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update(label.encode("utf-8") + b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    # Do not add a sentinel for legacy samples: their v2 semantic fingerprints
    # must remain exactly stable across this reader/schema upgrade.
    if sample.contact_trajectory is not None:
        array = np.ascontiguousarray(
            np.asarray(sample.contact_trajectory, dtype="<f8")
        )
        digest.update(b"contact_trajectory\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    digest.update(b"intercept_index\0")
    digest.update(np.asarray([sample.intercept_index], dtype="<i8").tobytes())
    return digest.hexdigest()


def feed_bank_content_hash(sample_fingerprints: list[str] | tuple[str, ...]) -> str:
    """Hash the ordered sample identities used by reset/feed indexing."""
    digest = hashlib.sha256()
    for fingerprint in sample_fingerprints:
        value = str(fingerprint)
        if len(value) != 64:
            raise ValueError("feed sample fingerprints must be SHA-256 hex digests")
        digest.update(value.encode("ascii") + b"\n")
    return digest.hexdigest()


def save_feed_bank(
    path: str | Path,
    bank: list[FeedSample],
    *,
    seed: int,
    cfg: FeedConfig | None = None,
    window: HitWindow | None = None,
    aero_cfg: ShuttlecockAeroConfig | None = None,
) -> Path:
    """Atomically persist an NPZ and its fail-closed provenance sidecar."""
    path = Path(path)
    if not bank:
        raise ValueError("cannot save an empty feed bank")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"n": np.array([len(bank)], dtype=np.int64)}
    fingerprints: list[str] = []
    for index, sample in enumerate(bank):
        fingerprints.append(feed_sample_fingerprint(sample))
        payload[f"launch_pos_{index}"] = np.asarray(sample.launch_pos, dtype=np.float64)
        payload[f"launch_vel_{index}"] = np.asarray(sample.launch_vel, dtype=np.float64)
        payload[f"trajectory_{index}"] = np.asarray(sample.trajectory, dtype=np.float64)
        if sample.contact_trajectory is not None:
            payload[f"contact_trajectory_{index}"] = np.asarray(
                sample.contact_trajectory, dtype=np.float64
            )
        payload[f"intercept_index_{index}"] = np.array(
            [sample.intercept_index], dtype=np.int64
        )

    npz_fd, npz_tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent
    )
    os.close(npz_fd)
    manifest_path = feed_bank_manifest_path(path)
    manifest_fd, manifest_tmp_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(manifest_fd)
    npz_tmp = Path(npz_tmp_name)
    manifest_tmp = Path(manifest_tmp_name)
    try:
        np.savez_compressed(npz_tmp, **payload)
        _fsync_file(npz_tmp)
        manifest = {
            **feed_bank_contract(
                seed=seed,
                sample_count=len(bank),
                cfg=cfg,
                window=window,
                aero_cfg=aero_cfg,
            ),
            "content_sha256": feed_bank_content_hash(fingerprints),
            "npz_sha256": _file_sha256(npz_tmp),
            "sample_fingerprints": fingerprints,
        }
        with manifest_tmp.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # If the process stops between these replaces, the old/new sidecar hash
        # mismatch makes the artifact unusable and the runner rebuilds it.
        os.replace(npz_tmp, path)
        os.replace(manifest_tmp, manifest_path)
        _fsync_directory(path.parent)
    finally:
        npz_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
    return path


def load_feed_bank_with_manifest(
    path: str | Path,
    *,
    expected_contract: Mapping[str, Any] | None = None,
) -> tuple[list[FeedSample], dict[str, Any]]:
    """Load only when sidecar, physical NPZ and semantic sample hashes agree."""
    path = Path(path)
    manifest = load_feed_bank_manifest(path, expected_contract=expected_contract)
    try:
        with np.load(path, allow_pickle=False) as payload:
            bank = _feed_bank_from_payload(payload, expected_count=manifest["sample_count"])
    except FeedBankValidationError:
        raise
    except Exception as exc:
        raise FeedBankValidationError(f"feed bank NPZ is unreadable: {path}") from exc

    fingerprints = [feed_sample_fingerprint(sample) for sample in bank]
    if fingerprints != manifest["sample_fingerprints"]:
        raise FeedBankValidationError("feed bank sample fingerprints differ from sidecar")
    content_hash = feed_bank_content_hash(fingerprints)
    if content_hash != manifest["content_sha256"]:
        raise FeedBankValidationError("feed bank semantic content hash differs from sidecar")
    return bank, manifest


def load_feed_bank(
    path: str | Path,
    *,
    expected_contract: Mapping[str, Any] | None = None,
) -> list[FeedSample]:
    bank, _manifest = load_feed_bank_with_manifest(path, expected_contract=expected_contract)
    return bank


def load_feed_bank_manifest(
    path: str | Path,
    *,
    expected_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    manifest_path = feed_bank_manifest_path(path)
    if not path.is_file() or not manifest_path.is_file():
        raise FeedBankValidationError(
            f"feed bank artifact is incomplete: {path} + {manifest_path.name}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeedBankValidationError(f"feed bank manifest is unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise FeedBankValidationError("feed bank manifest must contain a JSON object")

    required_fields = {
        "schema_version",
        "generator",
        "seed",
        "sample_count",
        "feed_config",
        "hit_window",
        "aero_config",
        "content_sha256",
        "npz_sha256",
        "sample_fingerprints",
    }
    if set(manifest) != required_fields:
        raise FeedBankValidationError(
            "feed bank manifest fields differ from the current schema"
        )
    if manifest.get("schema_version") != FEED_BANK_MANIFEST_SCHEMA:
        raise FeedBankValidationError("feed bank manifest schema version is unsupported")
    generator = manifest.get("generator")
    if not isinstance(generator, dict) or tuple(sorted(generator.items())) not in {
        tuple(sorted(FEED_BANK_GENERATOR.items())),
        tuple(sorted(CALIBRATED_FEED_BANK_GENERATOR.items())),
    }:
        raise FeedBankValidationError("feed bank generator identity/version changed")
    try:
        count = int(manifest["sample_count"])
    except (TypeError, ValueError) as exc:
        raise FeedBankValidationError("feed bank manifest sample_count is invalid") from exc
    if count <= 0 or count != manifest["sample_count"]:
        raise FeedBankValidationError("feed bank manifest sample_count must be a positive integer")
    fingerprints = manifest.get("sample_fingerprints")
    if not isinstance(fingerprints, list) or len(fingerprints) != count:
        raise FeedBankValidationError("feed bank manifest fingerprint count is invalid")
    for name in ("content_sha256", "npz_sha256"):
        value = manifest.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise FeedBankValidationError(f"feed bank manifest {name} is invalid")
    if _file_sha256(path) != manifest["npz_sha256"]:
        raise FeedBankValidationError("feed bank NPZ hash differs from sidecar")

    if expected_contract is not None:
        expected = _json_value(dict(expected_contract))
        actual = {key: manifest.get(key) for key in expected}
        if actual != expected:
            raise FeedBankValidationError("feed bank generation contract changed")
    return manifest


def _feed_bank_from_payload(payload: Any, *, expected_count: int) -> list[FeedSample]:
    if "n" not in payload.files:
        raise FeedBankValidationError("feed bank NPZ is missing n")
    n_array = np.asarray(payload["n"])
    if n_array.shape != (1,):
        raise FeedBankValidationError("feed bank NPZ n must have shape (1,)")
    n = int(n_array[0])
    if n != int(expected_count):
        raise FeedBankValidationError(
            f"feed bank NPZ count={n} differs from manifest={expected_count}"
        )
    expected_files = {"n"}
    for index in range(n):
        expected_files.update(
            {
                f"launch_pos_{index}",
                f"launch_vel_{index}",
                f"trajectory_{index}",
                f"intercept_index_{index}",
            }
        )
        contact_key = f"contact_trajectory_{index}"
        if contact_key in payload.files:
            expected_files.add(contact_key)
    if set(payload.files) != expected_files:
        raise FeedBankValidationError("feed bank NPZ fields differ from the current schema")

    bank: list[FeedSample] = []
    for index in range(n):
        trajectory = np.asarray(payload[f"trajectory_{index}"], dtype=np.float64)
        contact_key = f"contact_trajectory_{index}"
        contact_trajectory = (
            np.asarray(payload[contact_key], dtype=np.float64)
            if contact_key in payload.files
            else None
        )
        semantic_trajectory = (
            trajectory if contact_trajectory is None else contact_trajectory
        )
        index_array = np.asarray(payload[f"intercept_index_{index}"])
        if index_array.shape != (1,):
            raise FeedBankValidationError("feed bank intercept_index must have shape (1,)")
        intercept_index = int(index_array[0])
        sample = FeedSample(
            launch_pos=np.asarray(payload[f"launch_pos_{index}"], dtype=np.float64),
            launch_vel=np.asarray(payload[f"launch_vel_{index}"], dtype=np.float64),
            trajectory=trajectory,
            intercept_index=intercept_index,
            intercept_point=semantic_trajectory[intercept_index, 1:4].copy()
            if 0 <= intercept_index < len(semantic_trajectory)
            else np.empty((0,), dtype=np.float64),
            intercept_velocity=semantic_trajectory[intercept_index, 4:7].copy()
            if 0 <= intercept_index < len(semantic_trajectory)
            else np.empty((0,), dtype=np.float64),
            intercept_time_s=float(semantic_trajectory[intercept_index, 0])
            if 0 <= intercept_index < len(semantic_trajectory)
            else float("nan"),
            contact_trajectory=contact_trajectory,
        )
        _validate_feed_sample(sample)
        bank.append(sample)
    return bank


def _validate_feed_sample(sample: FeedSample) -> None:
    launch_pos = np.asarray(sample.launch_pos, dtype=float)
    launch_vel = np.asarray(sample.launch_vel, dtype=float)
    trajectory = np.asarray(sample.trajectory, dtype=float)
    if launch_pos.shape != (3,) or launch_vel.shape != (3,):
        raise FeedBankValidationError("feed sample launch position/velocity must have shape (3,)")
    if trajectory.ndim != 2 or trajectory.shape[1] != 7 or trajectory.shape[0] == 0:
        raise FeedBankValidationError("feed sample trajectory must have shape (T, 7), T>0")
    if not (
        np.isfinite(launch_pos).all()
        and np.isfinite(launch_vel).all()
        and np.isfinite(trajectory).all()
    ):
        raise FeedBankValidationError("feed sample contains non-finite values")
    contact_trajectory = (
        None
        if sample.contact_trajectory is None
        else np.asarray(sample.contact_trajectory, dtype=float)
    )
    if contact_trajectory is not None:
        if contact_trajectory.shape != trajectory.shape:
            raise FeedBankValidationError(
                "feed sample contact trajectory must match trajectory shape"
            )
        if not np.isfinite(contact_trajectory).all():
            raise FeedBankValidationError(
                "feed sample contact trajectory contains non-finite values"
            )
        if not np.array_equal(contact_trajectory[:, 0], trajectory[:, 0]):
            raise FeedBankValidationError(
                "feed sample contact trajectory timestamps differ from trajectory"
            )
    semantic_trajectory = (
        trajectory if contact_trajectory is None else contact_trajectory
    )
    index = int(sample.intercept_index)
    if not 0 <= index < trajectory.shape[0]:
        raise FeedBankValidationError("feed sample intercept_index is outside trajectory")
    if not np.array_equal(launch_pos, trajectory[0, 1:4]) or not np.array_equal(
        launch_vel, trajectory[0, 4:7]
    ):
        raise FeedBankValidationError("feed sample launch state differs from trajectory row zero")
    if np.asarray(sample.intercept_point).shape != (3,) or np.asarray(
        sample.intercept_velocity
    ).shape != (3,):
        raise FeedBankValidationError("feed sample intercept position/velocity must have shape (3,)")
    if not np.array_equal(
        np.asarray(sample.intercept_point), semantic_trajectory[index, 1:4]
    ):
        raise FeedBankValidationError("feed sample intercept point differs from trajectory")
    if not np.array_equal(
        np.asarray(sample.intercept_velocity), semantic_trajectory[index, 4:7]
    ):
        raise FeedBankValidationError("feed sample intercept velocity differs from trajectory")
    if not math.isclose(
        float(sample.intercept_time_s),
        float(semantic_trajectory[index, 0]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise FeedBankValidationError("feed sample intercept time differs from trajectory")


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError("feed bank provenance contains a non-finite float")
        return scalar
    if isinstance(value, (str, bool)) or value is None:
        return value
    raise TypeError(f"feed bank provenance cannot serialize {type(value).__name__}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
