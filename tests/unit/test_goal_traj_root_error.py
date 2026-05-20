import numpy as np

from musclemimic.core.goals.trajectory import _root_error_components


def test_root_error_components_aligns_reference_xy_to_episode_origin():
    sim_qpos = np.array([0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    sim_qvel = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ref_qpos = np.array([2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ref_qvel = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    init_ref_qpos = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    error = _root_error_components(sim_qpos, sim_qvel, ref_qpos, ref_qvel, init_ref_qpos, np)

    np.testing.assert_allclose(error[:3], np.array([0.8, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(error[3:9], np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(error[9:], np.array([0.0], dtype=np.float32), atol=1e-6)


def test_root_error_components_wraps_yaw_to_pi_interval():
    sim_qpos = np.array([0.0, 0.0, 1.0, -0.9998477, 0.0, 0.0, 0.0174524], dtype=np.float32)
    sim_qvel = np.zeros(6, dtype=np.float32)
    ref_qpos = np.array([0.0, 0.0, 1.0, -0.9998477, 0.0, 0.0, -0.0174524], dtype=np.float32)
    ref_qvel = np.zeros(6, dtype=np.float32)
    init_ref_qpos = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    error = _root_error_components(sim_qpos, sim_qvel, ref_qpos, ref_qvel, init_ref_qpos, np)

    assert abs(float(error[-1])) < 0.08
