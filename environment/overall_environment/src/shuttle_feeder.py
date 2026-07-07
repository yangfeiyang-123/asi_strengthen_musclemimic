"""Offline shuttle feed-trajectory generator for the incoming-hit task.

Integrates the drag-affected ballistic flight of a shuttlecock (pure numpy, no
MuJoCo data dependency) and rejection-samples launch states from the opposite
half court until the trajectory passes through the player's hit window.

The aerodynamic force reuses ``compute_shuttlecock_aero`` with the assumption
that the shuttle has already been righted by the pressure-center moment: the
nose axis tracks the incoming flow (angle of attack ~ 0), omega = 0, no wind.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class HitWindow:
    """Axis-aligned box in front of and above the player where a hit is possible."""

    x_range: tuple[float, float] = (-3.15, -2.25)
    y_range: tuple[float, float] = (-0.9, 0.9)
    z_range: tuple[float, float] = (1.2, 2.6)

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
    launch_x_range: tuple[float, float] = (2.5, 5.5)
    launch_y_range: tuple[float, float] = (-1.0, 1.0)
    launch_z_range: tuple[float, float] = (0.6, 1.2)
    speed_range: tuple[float, float] = (12.0, 22.0)
    elevation_deg_range: tuple[float, float] = (10.0, 45.0)
    azimuth_jitter_deg: float = 8.0
    integration_dt: float = 0.002
    max_flight_time: float = 3.0
    max_attempts: int = 200
    net_x: float = 0.0
    net_clearance_height: float = 1.60


@dataclass(frozen=True)
class FeedSample:
    launch_pos: np.ndarray
    launch_vel: np.ndarray
    trajectory: np.ndarray  # (T, 7): [t, x, y, z, vx, vy, vz]
    intercept_index: int
    intercept_point: np.ndarray
    intercept_velocity: np.ndarray
    intercept_time_s: float


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


def _crosses_net_cleanly(trajectory: np.ndarray, cfg: FeedConfig) -> bool:
    x = trajectory[:, 1]
    z = trajectory[:, 3]
    crossing = np.nonzero((x[:-1] > cfg.net_x) & (x[1:] <= cfg.net_x))[0]
    if crossing.size == 0:
        return False
    i = int(crossing[0])
    frac = (x[i] - cfg.net_x) / max(x[i] - x[i + 1], 1e-12)
    z_at_net = z[i] + frac * (z[i + 1] - z[i])
    return bool(z_at_net > cfg.net_clearance_height)


def sample_feed(
    rng: np.random.Generator,
    cfg: FeedConfig | None = None,
    window: HitWindow | None = None,
    aero_cfg: ShuttlecockAeroConfig | None = None,
) -> FeedSample:
    """Rejection-sample a launch state whose trajectory passes through the window."""
    if cfg is None:
        cfg = FeedConfig()
    if window is None:
        window = HitWindow()
    if aero_cfg is None:
        aero_cfg = ShuttlecockAeroConfig()

    for _ in range(cfg.max_attempts):
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
        to_center = window.center - launch_pos
        base_azimuth = float(np.arctan2(to_center[1], to_center[0]))
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
            continue
        inside = window.contains(trajectory[:, 1:4])
        approaching = trajectory[:, 4] < 0.0
        candidates = np.nonzero(inside & approaching)[0]
        if candidates.size == 0:
            continue
        intercept_index = int(candidates[0])
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
        f"failed to sample a feed reaching the hit window after {cfg.max_attempts} attempts"
    )


def build_feed_bank(
    n: int,
    seed: int,
    cfg: FeedConfig | None = None,
    window: HitWindow | None = None,
    aero_cfg: ShuttlecockAeroConfig | None = None,
) -> list[FeedSample]:
    rng = np.random.default_rng(seed)
    return [sample_feed(rng, cfg, window, aero_cfg) for _ in range(int(n))]


def save_feed_bank(path: str | Path, bank: list[FeedSample]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"n": np.array([len(bank)])}
    for index, sample in enumerate(bank):
        payload[f"launch_pos_{index}"] = sample.launch_pos
        payload[f"launch_vel_{index}"] = sample.launch_vel
        payload[f"trajectory_{index}"] = sample.trajectory
        payload[f"intercept_index_{index}"] = np.array([sample.intercept_index])
    np.savez_compressed(path, **payload)
    return path


def load_feed_bank(path: str | Path) -> list[FeedSample]:
    with np.load(Path(path)) as payload:
        n = int(payload["n"][0])
        bank = []
        for index in range(n):
            trajectory = payload[f"trajectory_{index}"]
            intercept_index = int(payload[f"intercept_index_{index}"][0])
            bank.append(
                FeedSample(
                    launch_pos=payload[f"launch_pos_{index}"],
                    launch_vel=payload[f"launch_vel_{index}"],
                    trajectory=trajectory,
                    intercept_index=intercept_index,
                    intercept_point=trajectory[intercept_index, 1:4].copy(),
                    intercept_velocity=trajectory[intercept_index, 4:7].copy(),
                    intercept_time_s=float(trajectory[intercept_index, 0]),
                )
            )
    return bank
