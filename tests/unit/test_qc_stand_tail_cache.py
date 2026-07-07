import numpy as np

from musclemimic.badminton.scripts.qc_stand_tail_cache import qc_cache


def _write_cache(path, *, com_xy=(0.0, 0.0), qvel_tail=0.0):
    frames = 6
    qpos = np.zeros((frames, 8), dtype=np.float32)
    qpos[:, 2] = 0.95
    qpos[:, 3] = 1.0
    qvel = np.zeros((frames, 7), dtype=np.float32)
    qvel[-2:] = qvel_tail
    site_names = np.asarray(
        [
            "pelvis_mimic",
            "left_ankle_mimic",
            "left_toes_mimic",
            "right_ankle_mimic",
            "right_toes_mimic",
        ]
    )
    site_xpos = np.zeros((frames, len(site_names), 3), dtype=np.float32)
    support = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-0.1, -0.1, 0.0],
            [-0.1, 0.2, 0.0],
            [0.1, -0.1, 0.0],
            [0.1, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    site_xpos[:] = support
    subtree_com = np.zeros((frames, 2, 3), dtype=np.float32)
    subtree_com[:, 1, :2] = np.asarray(com_xy, dtype=np.float32)
    subtree_com[:, 1, 2] = 0.95
    np.savez_compressed(
        path,
        qpos=qpos,
        qvel=qvel,
        site_xpos=site_xpos,
        site_names=site_names,
        subtree_com=subtree_com,
        frequency=np.asarray(100.0),
        metadata={
            "original_frames": 3,
            "settle_frames": 1,
            "hold_frames": 2,
        },
    )


def test_qc_cache_passes_static_supported_tail(tmp_path) -> None:
    cache = tmp_path / "motion.npz"
    _write_cache(cache)

    row = qc_cache(
        cache,
        "motion",
        max_tail_qvel=1e-6,
        max_tail_qpos_step=1e-6,
        min_root_height=0.6,
        max_com_support_margin=0.2,
    )

    assert row.passed
    assert row.tail_qvel_max == 0.0
    assert row.com_support_margin_max_m == 0.0


def test_qc_cache_flags_tail_velocity_and_com_margin(tmp_path) -> None:
    cache = tmp_path / "motion.npz"
    _write_cache(cache, com_xy=(1.0, 1.0), qvel_tail=0.1)

    row = qc_cache(
        cache,
        "motion",
        max_tail_qvel=1e-6,
        max_tail_qpos_step=1e-6,
        min_root_height=0.6,
        max_com_support_margin=0.2,
    )

    assert not row.passed
    assert "tail_qvel_max" in row.failures
    assert "com_support_margin_max" in row.failures
