import numpy as np

from musclemimic.badminton.scripts.filter_retarget_cache import _velocity_limit_columns, filter_qpos


def test_velocity_limit_columns_caps_selected_deltas_only():
    values = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [20.0, -10.0, 20.0],
        ],
        dtype=float,
    )

    filtered = _velocity_limit_columns(values, np.array([1]), max_step=2.0)

    np.testing.assert_allclose(filtered[:, 0], values[:, 0])
    np.testing.assert_allclose(filtered[:, 2], values[:, 2])
    np.testing.assert_allclose(filtered[:, 1], [0.0, 2.0, 0.0])


def test_filter_qpos_limits_scalar_joint_speed_and_normalizes_root_quat():
    qpos = np.zeros((5, 9), dtype=float)
    qpos[:, 3] = 2.0
    qpos[:, 7] = [0.0, 1.0, 2.0, -2.0, 4.0]
    qpos[:, 8] = [0.0, -1.0, -2.0, 2.0, -4.0]

    filtered = filter_qpos(
        qpos,
        frequency=100.0,
        joint_sigma=0.0,
        max_joint_speed=5.0,
        root_sigma=0.0,
        max_root_speed=2.0,
    )

    np.testing.assert_allclose(np.linalg.norm(filtered[:, 3:7], axis=1), np.ones(5))
    assert np.max(np.abs(np.diff(filtered[:, 7:], axis=0))) <= 0.050001
