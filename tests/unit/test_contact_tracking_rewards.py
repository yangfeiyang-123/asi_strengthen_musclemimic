import numpy as np

from musclemimic.badminton.asi.rewards import compute_body_graph_laplacian_error, compute_contact_tracking_components


def test_body_graph_laplacian_error_is_zero_for_matching_targets():
    current = np.zeros((2, 3), dtype=np.float32)
    reference = np.zeros((2, 3), dtype=np.float32)

    assert compute_body_graph_laplacian_error(current, reference) == 0.0


def test_contact_tracking_components_report_body_and_stance_terms():
    reference_body_laplacian = np.zeros((2, 3), dtype=np.float32)
    actual_body_laplacian = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    reference_feet = np.zeros((3, 1, 3), dtype=np.float32)
    actual_feet = reference_feet.copy()
    actual_feet[1, 0, 2] = 0.10
    actual_feet[2, 0, 0] = 0.10
    stance_mask = np.ones((3, 1), dtype=np.bool_)

    components = compute_contact_tracking_components(
        actual_body_laplacian=actual_body_laplacian,
        reference_body_laplacian=reference_body_laplacian,
        actual_foot_points=actual_feet,
        reference_foot_points=reference_feet,
        stance_mask=stance_mask,
        fps=60.0,
    )

    assert components["body_graph_error"] > 0.0
    assert components["foot_contact_height_error"] > 0.0
    assert components["foot_contact_speed_mps"] > 0.0
    assert 0.0 < components["reward_body_graph"] < 1.0
    assert 0.0 < components["reward_foot_contact_height"] < 1.0
