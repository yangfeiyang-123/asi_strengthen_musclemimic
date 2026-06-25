import numpy as np
import pytest
from pathlib import Path


def _make_fake_cache(tmp_path: Path, num_frames: int = 60, num_feet: int = 4) -> Path:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.savez(
        cache_dir / "tracking_reference_cache.npz",
        poses_ref=np.zeros((num_frames, 72), dtype=np.float32),
        trans_ref=np.zeros((num_frames, 3), dtype=np.float32),
        contact_confidence=np.random.rand(num_frames, num_feet).astype(np.float32),
        stance_mask=np.random.rand(num_frames, num_feet) > 0.5,
        foot_points=np.random.randn(num_frames, num_feet, 3).astype(np.float32),
        foot_labels=np.array(["left_ankle", "left_toe", "right_ankle", "right_toe"]),
        reference_fps=np.float32(60.0),
        control_dt=np.float32(0.01),
        effective_ref_stride=np.float32(0.6),
        coordinate_system=np.array("amass_zup"),
        source_manifest=np.array("/fake/manifest.json"),
    )
    return cache_dir


def test_load_from_cache_dir(tmp_path):
    from BadmintonMimic.asi.contact_tracking_data import load_contact_tracking_data

    cache_dir = _make_fake_cache(tmp_path, num_frames=60, num_feet=4)
    ctd = load_contact_tracking_data(cache_dir, control_dt=0.01)
    assert ctd.stance_mask.shape == (60, 4)
    assert ctd.foot_points.shape == (60, 4, 3)
    assert ctd.num_frames == 60
    assert len(ctd.foot_labels) == 4


def test_frame_at_traj_step(tmp_path):
    from BadmintonMimic.asi.contact_tracking_data import load_contact_tracking_data

    cache_dir = _make_fake_cache(tmp_path, num_frames=100)
    ctd = load_contact_tracking_data(cache_dir, control_dt=0.01)
    assert ctd.frame_at_traj_step(0) == 0
    assert 0 <= ctd.frame_at_traj_step(999) < ctd.num_frames


def test_missing_cache_raises(tmp_path):
    from BadmintonMimic.asi.contact_tracking_data import load_contact_tracking_data

    with pytest.raises(FileNotFoundError):
        load_contact_tracking_data(tmp_path / "nonexistent", control_dt=0.01)
