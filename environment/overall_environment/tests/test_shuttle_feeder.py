from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.shuttle_feeder import (  # noqa: E402
    CALIBRATED_FEED_BANK_GENERATOR,
    FEED_BANK_GENERATOR,
    FEED_BANK_MANIFEST_SCHEMA,
    FeedBankValidationError,
    FeedConfig,
    FeedSample,
    HitWindow,
    build_feed_bank,
    feed_bank_contract,
    feed_sample_fingerprint,
    feed_bank_quality_report,
    feed_bank_manifest_path,
    integrate_shuttle_flight,
    launch_quat_from_velocity,
    load_feed_bank,
    load_feed_bank_with_manifest,
    sample_feed,
    save_feed_bank,
    net_crossing_height,
    render_feed_bank_qc,
)


def test_integrate_reaches_terminal_velocity() -> None:
    trajectory = integrate_shuttle_flight(
        np.array([0.0, 0.0, 30.0]), np.zeros(3), dt=0.002, t_max=4.0
    )
    final_speed = float(np.linalg.norm(trajectory[-1, 4:7]))
    assert 6.5 <= final_speed <= 7.2  # nominal vt = 6.86 m/s


def test_sample_feed_intercepts_window() -> None:
    rng = np.random.default_rng(11)
    cfg = FeedConfig()
    window = HitWindow()
    for _ in range(20):
        sample = sample_feed(rng, cfg, window)
        assert window.contains(sample.intercept_point[None])[0]
        assert sample.intercept_velocity[0] < 0.0
        assert cfg.intercept_vertical_velocity_range_m_s[0] <= sample.intercept_velocity[2]
        assert sample.intercept_velocity[2] <= cfg.intercept_vertical_velocity_range_m_s[1]
        assert sample.launch_pos[0] >= cfg.launch_x_range[0]
        assert cfg.intercept_time_range_s[0] <= sample.intercept_time_s
        assert sample.intercept_time_s <= cfg.intercept_time_range_s[1]
        assert cfg.apex_height_range_m[0] <= sample.trajectory[:, 3].max()
        assert sample.trajectory[:, 3].max() <= cfg.apex_height_range_m[1]
        assert net_crossing_height(sample.trajectory, cfg) > cfg.net_clearance_height


def test_feed_bank_quality_has_safe_height_coverage_and_renders(tmp_path: Path) -> None:
    cfg = FeedConfig()
    window = HitWindow()
    bank = build_feed_bank(32, seed=17, cfg=cfg, window=window)
    report = feed_bank_quality_report(bank, cfg, window)
    assert report["schema_version"] == "incoming_shuttle_feed_quality_v2"
    assert report["distribution_gates_applicable"] is True
    assert report["passed"] is True
    assert all(report["gates"].values())
    assert report["intercept_height_coverage_fraction_p05_p95"] >= 0.65
    output = render_feed_bank_qc(tmp_path / "feed_qc.png", bank, cfg, window)
    assert output.is_file()
    assert output.stat().st_size > 0


def test_intercept_selection_targets_height_instead_of_first_window_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import environment.overall_environment.src.shuttle_feeder as feeder

    seed = 41
    probe = np.random.default_rng(seed)
    probe.uniform(*HitWindow().x_range)
    probe.uniform(*HitWindow().y_range)
    target_z = probe.uniform(*HitWindow().z_range)

    def fake_integrate(pos0, vel0, **_kwargs):
        return np.asarray(
            [
                [0.0, *pos0, *vel0],
                [0.50, 0.20, 0.0, 3.00, -9.0, 0.0, 1.0],
                [0.60, -0.20, 0.0, 3.10, -9.0, 0.0, -1.0],
                [1.10, -2.90, 0.0, 2.24, -8.0, 0.0, -3.0],
                [1.20, -2.70, 0.0, target_z, -8.0, 0.0, -3.0],
            ],
            dtype=float,
        )

    monkeypatch.setattr(feeder, "integrate_shuttle_flight", fake_integrate)
    sample = sample_feed(np.random.default_rng(seed), FeedConfig(max_attempts=1), HitWindow())
    assert sample.intercept_index == 4
    assert sample.intercept_point[2] == pytest.approx(target_z)


def test_feed_contract_rejects_invalid_overhead_ranges() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        HitWindow(z_range=(2.0, 2.0))
    with pytest.raises(ValueError, match="descending"):
        FeedConfig(intercept_vertical_velocity_range_m_s=(-1.0, 1.0))


def test_feed_bank_deterministic_and_roundtrip(tmp_path: Path) -> None:
    bank_a = build_feed_bank(4, seed=3)
    bank_b = build_feed_bank(4, seed=3)
    for a, b in zip(bank_a, bank_b):
        np.testing.assert_allclose(a.launch_pos, b.launch_pos)
        np.testing.assert_allclose(a.launch_vel, b.launch_vel)

    cfg = FeedConfig()
    window = HitWindow()
    path = save_feed_bank(
        tmp_path / "bank.npz", bank_a, seed=3, cfg=cfg, window=window
    )
    loaded, manifest = load_feed_bank_with_manifest(
        path,
        expected_contract=feed_bank_contract(
            seed=3, sample_count=4, cfg=cfg, window=window
        ),
    )
    assert manifest["schema_version"] == FEED_BANK_MANIFEST_SCHEMA
    assert manifest["sample_count"] == 4
    assert manifest["seed"] == 3
    assert len(manifest["sample_fingerprints"]) == 4
    assert len(manifest["content_sha256"]) == 64
    assert len(manifest["npz_sha256"]) == 64
    assert feed_bank_manifest_path(path).is_file()
    assert len(loaded) == len(bank_a)
    for a, b in zip(bank_a, loaded):
        np.testing.assert_allclose(a.trajectory, b.trajectory)
        assert a.intercept_index == b.intercept_index
        np.testing.assert_allclose(a.intercept_point, b.intercept_point)
        assert b.contact_trajectory is None


def test_contact_trajectory_roundtrip_and_legacy_fingerprint_stability(
    tmp_path: Path,
) -> None:
    trajectory = np.asarray(
        [
            [0.0, 3.0, 0.0, 2.0, -9.0, 0.0, 17.0],
            [0.1, 2.1, 0.0, 3.7, -8.8, 0.0, 16.0],
        ],
        dtype=float,
    )
    cork = trajectory.copy()
    cork[:, 1:4] += np.asarray([0.01, -0.02, 0.03])
    legacy = FeedSample(
        launch_pos=trajectory[0, 1:4].copy(),
        launch_vel=trajectory[0, 4:7].copy(),
        trajectory=trajectory,
        intercept_index=1,
        intercept_point=trajectory[1, 1:4].copy(),
        intercept_velocity=trajectory[1, 4:7].copy(),
        intercept_time_s=float(trajectory[1, 0]),
    )
    # This digest was produced by the v2 algorithm before contact_trajectory
    # existed.  Keeping it fixed protects every persisted legacy manifest.
    assert feed_sample_fingerprint(legacy) == (
        "57198da3781948dcb00800ceaeabe848011a086edfd87e1a64c3b6cc0c8377d7"
    )

    sample = FeedSample(
        launch_pos=trajectory[0, 1:4].copy(),
        launch_vel=trajectory[0, 4:7].copy(),
        trajectory=trajectory,
        intercept_index=1,
        intercept_point=cork[1, 1:4].copy(),
        intercept_velocity=cork[1, 4:7].copy(),
        intercept_time_s=float(cork[1, 0]),
        contact_trajectory=cork,
    )
    cfg = FeedConfig()
    path = save_feed_bank(
        tmp_path / "contact_bank.npz", [sample], seed=7, cfg=cfg
    )
    loaded = load_feed_bank(
        path,
        expected_contract=feed_bank_contract(seed=7, sample_count=1, cfg=cfg),
    )
    assert loaded[0].contact_trajectory is not None
    np.testing.assert_array_equal(loaded[0].trajectory, trajectory)
    np.testing.assert_array_equal(loaded[0].contact_trajectory, cork)
    np.testing.assert_array_equal(loaded[0].intercept_point, cork[1, 1:4])


def test_calibrated_curriculum_is_training_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import environment.overall_environment.src.shuttle_feeder as feeder

    scales: list[float] = []

    class _FakeIntegrator:
        def __init__(self, **_kwargs) -> None:
            pass

    def fake_sample(*, jitter_scale: float, **_kwargs):
        scales.append(jitter_scale)
        trajectory = np.asarray(
            [[0.0, 3.0, 0.0, 2.0, -9.0, 0.0, 17.0]], dtype=float
        )
        return feeder.FeedSample(
            launch_pos=trajectory[0, 1:4],
            launch_vel=trajectory[0, 4:7],
            trajectory=trajectory,
            intercept_index=0,
            intercept_point=np.asarray([-2.5, -0.1, 2.04]),
            intercept_velocity=np.asarray([-1.0, 0.0, -6.2]),
            intercept_time_s=1.88,
        )

    monkeypatch.setattr(feeder, "_RigidBodyFlightIntegrator", _FakeIntegrator)
    monkeypatch.setattr(feeder, "_calibrated_sample", fake_sample)
    cfg = FeedConfig(
        sampling_mode="calibrated_rigid_body_v3",
        intercept_time_range_s=(1.80, 1.96),
        calibrated_reference_seed=17,
        calibrated_warmup_count=2,
    )

    build_feed_bank(34, seed=17, cfg=cfg)
    assert scales[0] == 0.0
    assert scales[1] == 0.15
    assert scales[2:32] == [0.45] * 30
    assert scales[32:] == [1.0, 1.0]

    scales.clear()
    build_feed_bank(8, seed=1017, cfg=cfg)
    assert scales == [1.0] * 8


def test_legacy_feed_contract_ignores_v3_only_calibration_fields() -> None:
    baseline = feed_bank_contract(
        seed=5,
        sample_count=2,
        cfg=FeedConfig(),
        window=HitWindow(),
    )
    calibration_drift = feed_bank_contract(
        seed=5,
        sample_count=2,
        cfg=FeedConfig(
            calibrated_intercept_time_s=1.91,
            calibrated_intercept_fraction_jitter=0.31,
            calibrated_warmup_count=7,
        ),
        window=HitWindow(),
    )
    assert baseline == calibration_drift
    assert baseline["generator"] == FEED_BANK_GENERATOR

    calibrated = feed_bank_contract(
        seed=5,
        sample_count=2,
        cfg=FeedConfig(
            sampling_mode="calibrated_rigid_body_v3",
            intercept_time_range_s=(1.80, 1.96),
        ),
        window=HitWindow(),
    )
    assert calibrated["generator"] == CALIBRATED_FEED_BANK_GENERATOR


def test_feed_bank_rejects_missing_manifest_contract_drift_and_npz_tamper(
    tmp_path: Path,
) -> None:
    bank = build_feed_bank(2, seed=5)
    path = tmp_path / "bank.npz"

    # Legacy NPZ-only caches are deliberately not trusted.
    np.savez_compressed(path, n=np.asarray([0]))
    with pytest.raises(FeedBankValidationError, match="incomplete"):
        load_feed_bank(path)

    save_feed_bank(path, bank, seed=5, cfg=FeedConfig(), window=HitWindow())
    drifted = feed_bank_contract(
        seed=5,
        sample_count=2,
        cfg=FeedConfig(azimuth_jitter_deg=3.0),
        window=HitWindow(),
    )
    with pytest.raises(FeedBankValidationError, match="contract changed"):
        load_feed_bank(path, expected_contract=drifted)

    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(FeedBankValidationError, match="NPZ hash"):
        load_feed_bank(path)


def test_rejection_raises_on_impossible_window() -> None:
    rng = np.random.default_rng(0)
    impossible = HitWindow(x_range=(-3.0, -2.5), y_range=(-0.5, 0.5), z_range=(-2.0, -1.0))
    cfg = FeedConfig(max_attempts=20)
    with pytest.raises(RuntimeError):
        sample_feed(rng, cfg, impossible)


def test_launch_quat_aligns_nose_with_velocity() -> None:
    vel = np.array([-8.0, 1.5, 3.0])
    quat = launch_quat_from_velocity(vel)
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-9)
    w, x, y, z = quat
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    nose_world = rot @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(nose_world, vel / np.linalg.norm(vel), atol=1e-9)
