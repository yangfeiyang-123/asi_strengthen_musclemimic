"""Offline shuttle feed-trajectory generator for the incoming-hit task.

Integrates the drag-affected ballistic flight of a shuttlecock (pure numpy, no
MuJoCo data dependency) and rejection-samples launch states from the opposite
half court until the trajectory passes through the player's hit window.

The aerodynamic force reuses ``compute_shuttlecock_aero`` with the assumption
that the shuttle has already been righted by the pressure-center moment: the
nose axis tracks the incoming flow (angle of attack ~ 0), omega = 0, no wind.
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
    "version": "1",
}


class FeedBankValidationError(ValueError):
    """Raised when a persisted feed bank does not match its exact provenance."""


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
    return {
        "schema_version": FEED_BANK_MANIFEST_SCHEMA,
        "generator": dict(FEED_BANK_GENERATOR),
        "seed": int(seed),
        "sample_count": int(sample_count),
        "feed_config": _json_value(FeedConfig() if cfg is None else cfg),
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
    if manifest.get("generator") != FEED_BANK_GENERATOR:
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
    if set(payload.files) != expected_files:
        raise FeedBankValidationError("feed bank NPZ fields differ from the current schema")

    bank: list[FeedSample] = []
    for index in range(n):
        trajectory = np.asarray(payload[f"trajectory_{index}"], dtype=np.float64)
        index_array = np.asarray(payload[f"intercept_index_{index}"])
        if index_array.shape != (1,):
            raise FeedBankValidationError("feed bank intercept_index must have shape (1,)")
        intercept_index = int(index_array[0])
        sample = FeedSample(
            launch_pos=np.asarray(payload[f"launch_pos_{index}"], dtype=np.float64),
            launch_vel=np.asarray(payload[f"launch_vel_{index}"], dtype=np.float64),
            trajectory=trajectory,
            intercept_index=intercept_index,
            intercept_point=trajectory[intercept_index, 1:4].copy()
            if 0 <= intercept_index < len(trajectory)
            else np.empty((0,), dtype=np.float64),
            intercept_velocity=trajectory[intercept_index, 4:7].copy()
            if 0 <= intercept_index < len(trajectory)
            else np.empty((0,), dtype=np.float64),
            intercept_time_s=float(trajectory[intercept_index, 0])
            if 0 <= intercept_index < len(trajectory)
            else float("nan"),
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
    if not np.array_equal(np.asarray(sample.intercept_point), trajectory[index, 1:4]):
        raise FeedBankValidationError("feed sample intercept point differs from trajectory")
    if not np.array_equal(np.asarray(sample.intercept_velocity), trajectory[index, 4:7]):
        raise FeedBankValidationError("feed sample intercept velocity differs from trajectory")
    if not math.isclose(
        float(sample.intercept_time_s), float(trajectory[index, 0]), rel_tol=0.0, abs_tol=0.0
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
