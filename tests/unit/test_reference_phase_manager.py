import numpy as np

from BadmintonMimic.asi.reference_phase import ReferencePhaseManager


def test_reference_phase_manager_maps_control_steps_to_reference_frames():
    manager = ReferencePhaseManager(num_frames=6, reference_fps=60.0, control_dt=1.0 / 30.0)

    assert manager.effective_ref_stride == 2.0
    assert [manager.frame_at_control_step(step) for step in range(5)] == [0, 2, 4, 5, 5]


def test_reference_phase_manager_samples_future_indices_with_clamping():
    manager = ReferencePhaseManager(num_frames=5, reference_fps=60.0, control_dt=1.0 / 60.0)

    assert np.array_equal(manager.sample_indices(start_frame=3, offsets=[0, 1, 3]), np.array([3, 4, 4], dtype=np.int32))
