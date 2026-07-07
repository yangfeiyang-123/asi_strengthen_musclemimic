import numpy as np

from musclemimic.badminton.asi.diagnostics import evaluate_tracking_diagnostics


def test_tracking_diagnostics_reports_root_error_and_stance_slip_frames():
    reference_trans = np.zeros((4, 3), dtype=np.float32)
    actual_trans = reference_trans.copy()
    actual_trans[2, 0] = 1.0
    foot_points = np.zeros((4, 1, 3), dtype=np.float32)
    foot_points[2, 0, 0] = 0.20
    foot_points[3, 0, 0] = 0.20
    stance_mask = np.ones((4, 1), dtype=np.bool_)

    report = evaluate_tracking_diagnostics(
        reference_trans=reference_trans,
        actual_trans=actual_trans,
        actual_foot_points=foot_points,
        stance_mask=stance_mask,
        fps=60.0,
        root_error_threshold_m=0.5,
        stance_speed_threshold_mps=5.0,
    )

    failures = {(item["frame"], item["failure"]) for item in report["failed_frames"]}

    assert (2, "root_tracking_error") in failures
    assert (2, "stance_foot_slip") in failures
    assert report["summary"]["num_failed_frames"] == 1
